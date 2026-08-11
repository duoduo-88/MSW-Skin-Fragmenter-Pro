import multiprocessing as mp
import secrets
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QThread, Signal

from .degrade import degrade_chunk_worker, simple_block_degrade
from .fragmentation import (
    apply_overlap_to_all_fragments,
    split_fragments,
    split_fragments_with_secondary_overflow,
)
from .interference import build_interfere_block_pool, generate_interference_worker


class SplitThread(QtCore.QThread):
    update_progress = QtCore.Signal(int, int, str)
    result = QtCore.Signal(list)

    def __init__(
        self,
        main_image,
        mask_image,
        count,
        block_size,
        random_factor,
        strict_mask=True,
        secondary_overflow_mask=None,
        seed=None,
    ):
        super().__init__()
        self.main_image = main_image.copy()
        # 次要溢出流程不會使用一般 mask_image，避免同一遮罩重複複製。
        self.mask_image = (
            mask_image.copy()
            if mask_image is not None and secondary_overflow_mask is None
            else None
        )
        self.count = count
        self.block_size = block_size
        self.random_factor = random_factor
        self.strict_mask = strict_mask
        self.secondary_overflow_mask = (
            secondary_overflow_mask.copy()
            if secondary_overflow_mask is not None
            else None
        )
        self.seed = int(seed) if seed is not None else secrets.randbits(64)
        self._abort_event = threading.Event()
        self._last_time = 0.0

    def abort(self):
        self._abort_event.set()

    def run(self):
        started = time.time()
        try:
            if self.secondary_overflow_mask is not None:
                fragments = split_fragments_with_secondary_overflow(
                    self.main_image,
                    self.secondary_overflow_mask,
                    self.count,
                    self.block_size,
                    self.random_factor,
                    progress_cb=self.update_progress.emit,
                    abort_cb=self._abort_event.is_set,
                    seed=self.seed,
                )
            else:
                fragments = split_fragments(
                    self.main_image,
                    self.mask_image,
                    self.count,
                    self.block_size,
                    self.random_factor,
                    strict_mask=self.strict_mask,
                    progress_cb=self.update_progress.emit,
                    abort_cb=self._abort_event.is_set,
                    seed=self.seed,
                )
            self._last_time = time.time() - started
            if not self._abort_event.is_set():
                self.result.emit(fragments)
        except Exception as exc:
            print(f"[SplitThread Exception]: {exc}")
            if not self._abort_event.is_set():
                self.result.emit([])


class OverlapThread(QThread):
    progress = Signal(int, int, str)
    result = Signal(list)

    def __init__(
        self,
        images,
        main_image,
        mask_image,
        fill_percent,
        block_size,
        random_range,
        aggregation=1,
        limit_to_mask=False,
    ):
        super().__init__()
        # overlap worker 會建立輸出副本；此處只保留參照，避免先複製整套碎片。
        self.images = list(images)
        self.main_image = main_image
        self.mask_image = mask_image
        self.fill_percent = fill_percent
        self.block_size = block_size
        self.random_range = random_range
        self.aggregation = aggregation
        self.limit_to_mask = limit_to_mask
        self._abort_event = threading.Event()

    def abort(self):
        self._abort_event.set()

    def run(self):
        try:
            result = apply_overlap_to_all_fragments(
                self.images,
                self.main_image,
                self.mask_image,
                self.fill_percent,
                self.block_size,
                self.random_range,
                limit_to_mask=self.limit_to_mask,
                aggregation=self.aggregation,
                progress_cb=self.progress.emit,
                abort_cb=self._abort_event.is_set,
            )
            if not self._abort_event.is_set():
                self.result.emit(result)
        except Exception as exc:
            print(f"[OverlapThread Exception]: {exc}")
            if not self._abort_event.is_set():
                self.result.emit([])


class InterfereGenThread(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)
    result = QtCore.Signal(dict)

    def __init__(
        self, fragment_data, settings, fragment_order, base_source, seed=None
    ):
        super().__init__()
        self.fragment_data = fragment_data
        self.settings = settings
        self.fragment_order = list(fragment_order)
        self.base_source = base_source
        self.seed = int(seed) if seed is not None else secrets.randbits(64)
        self._abort_event = threading.Event()
        self._executor = None
        self._used_thread_pool = False

    def abort(self):
        # executor 只由 run() 所在執行緒關閉，避免和 submit/as_completed 競態。
        self._abort_event.set()

    def run(self):
        started = time.time()
        result = {}
        try:
            seed_sequence = np.random.SeedSequence(self.seed)
            child_sequences = seed_sequence.spawn(max(2, len(self.fragment_order) + 1))
            self.progress.emit(0, max(1, len(self.fragment_order) - 2), "建立共用主圖素材池...")
            base_pool = build_interfere_block_pool(
                self.base_source,
                self.settings["block_size"],
                self.settings["random_range"],
                pool_size=300,
                alpha_min=self.settings.get("alpha_min", 1),
                alpha_max=self.settings.get("alpha_max", 100),
                # 主圖只負責提供干擾像素；生成範圍由每片的前方碎片決定。
                restrict_mask=None,
                rng=np.random.default_rng(child_sequences[0]),
                abort_cb=self._abort_event.is_set,
            )
            if self._abort_event.is_set():
                return
            if not base_pool:
                self.result.emit({})
                return

            args_list = []
            for index, name in enumerate(self.fragment_order):
                # 第一張既不加干擾，也不能提供範圍；第二張沒有可用的前方範圍。
                # 因此從第三張開始，候選範圍只取第二張到目前碎片的前一張。
                if index < 2 or name not in self.fragment_data:
                    continue
                scope_names = self.fragment_order[1:index]
                if not self.settings.get("random_previous_scopes", True):
                    scope_names = self.fragment_order[1:2]
                scope_candidates = [
                    (previous_name, self.fragment_data[previous_name])
                    for previous_name in scope_names
                    if previous_name in self.fragment_data
                ]
                args_list.append(
                    (
                        name,
                        self.fragment_data[name],
                        scope_candidates,
                        base_pool,
                        self.settings,
                        int(child_sequences[index + 1].generate_state(1, dtype=np.uint64)[0]),
                        self._abort_event,
                    )
                )

            total = len(args_list)
            if total == 0:
                self.result.emit({})
                return

            completed = 0

            def accept_worker_result(name, worker_result):
                nonlocal completed
                try:
                    result_name, interference, selected_scope_names = worker_result
                    result[result_name] = interference
                    scope_text = " + ".join(selected_scope_names) or "無可用範圍"
                    self.progress.emit(
                        completed,
                        total,
                        f"{name} 的干擾範圍：{scope_text}",
                    )
                except Exception as exc:
                    print(f"產生 {name} 干擾像素失敗: {exc}")
                completed += 1
                elapsed = time.time() - started
                self.progress.emit(completed, total, f"產生干擾像素中，已用 {elapsed:.1f} 秒")

            image_pixels = int(self.base_source.shape[0] * self.base_source.shape[1])
            workload = image_pixels * total
            thread_threshold = int(
                self.settings.get("thread_workload_threshold", 1_000_000)
            )
            # Small patches spend most of their time in Python control flow and
            # become slower under the GIL. Larger NumPy patch composites release
            # enough of the GIL for the bounded thread pool to be worthwhile.
            thread_min_block_size = int(
                self.settings.get("thread_min_block_size", 8)
            )
            self._used_thread_pool = (
                total > 1
                and self.settings.get("block_size", 1) >= thread_min_block_size
                and workload >= thread_threshold
            )

            if not self._used_thread_pool:
                for args in args_list:
                    if self._abort_event.is_set():
                        return
                    try:
                        worker_result = generate_interference_worker(args)
                    except Exception as exc:
                        print(f"產生 {args[0]} 干擾像素失敗: {exc}")
                        completed += 1
                        continue
                    accept_worker_result(args[0], worker_result)
            else:
                # 大工作才使用執行緒；共用唯讀 RGBA 與素材池，避免程序複製。
                self._executor = ThreadPoolExecutor(max_workers=min(4, total))
                futures = {
                    self._executor.submit(generate_interference_worker, args): args[0]
                    for args in args_list
                }
                for future in as_completed(futures):
                    if self._abort_event.is_set():
                        for pending in futures:
                            pending.cancel()
                        return
                    name = futures[future]
                    try:
                        worker_result = future.result()
                    except Exception as exc:
                        print(f"產生 {name} 干擾像素失敗: {exc}")
                        completed += 1
                        continue
                    accept_worker_result(name, worker_result)
            if self._abort_event.is_set():
                return
            self.result.emit(result)
        except Exception as exc:
            print(f"[InterfereGenThread Exception]: {exc}")
            if not self._abort_event.is_set():
                self.result.emit({})
        finally:
            if self._executor is not None:
                self._executor.shutdown(
                    wait=True,
                    cancel_futures=self._abort_event.is_set(),
                )
                self._executor = None


class DegradePreviewThread(QtCore.QThread):
    progress = Signal(int, int, str)
    result = Signal(np.ndarray)

    def __init__(self, source, settings):
        super().__init__()
        self.source = source.copy()
        self.settings = settings
        self._abort_event = threading.Event()
        self._executor = None

    def abort(self):
        # The worker thread owns the executor. Shutting it down from the UI
        # thread can race with submit() and as_completed().
        self._abort_event.set()

    def run(self):
        try:
            block_size = self.settings["block_size"]
            density = self.settings["density"]
            h, w = self.source.shape[:2]
            estimated_blocks = h * w * density / max(1, block_size * block_size)
            max_workers = min(mp.cpu_count(), 4)
            if estimated_blocks < 50 or max_workers <= 1:
                degraded = simple_block_degrade(
                    self.source,
                    block_size,
                    self.settings["rand_range"],
                    density,
                    self.settings["noise_strength"],
                    self.settings["brightness_strength"],
                    self.settings["color_strength"],
                )
                if not self._abort_event.is_set():
                    self.result.emit(degraded)
                return

            boundaries = np.linspace(0, w, max_workers + 1, dtype=int)
            tasks = []
            for index in range(max_workers):
                x0, x1 = boundaries[index], boundaries[index + 1]
                tasks.append(((self.source[:, x0:x1].copy(), self.settings), x0, x1))

            out = np.zeros_like(self.source)
            self._executor = ProcessPoolExecutor(max_workers=max_workers)
            futures = {}
            for chunk_args, x0, x1 in tasks:
                if self._abort_event.is_set():
                    return
                future = self._executor.submit(degrade_chunk_worker, chunk_args)
                futures[future] = (x0, x1)
            completed = 0
            for future in as_completed(futures):
                if self._abort_event.is_set():
                    return
                x0, x1 = futures[future]
                try:
                    out[:, x0:x1] = future.result()
                except Exception:
                    out[:, x0:x1] = self.source[:, x0:x1]
                completed += 1
                self.progress.emit(completed, len(tasks), "劣化中...")
            out[..., 3] = self.source[..., 3]
            if not self._abort_event.is_set():
                self.result.emit(out)
        except Exception as exc:
            print(f"[DegradePreviewThread Exception]: {exc}")
            if not self._abort_event.is_set():
                self.result.emit(self.source)
        finally:
            if self._executor is not None:
                self._executor.shutdown(
                    wait=True,
                    cancel_futures=self._abort_event.is_set(),
                )
                self._executor = None
