import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from PIL import Image


def pil2np(image):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return np.array(image)


def np2pil(array):
    return Image.fromarray(array.astype(np.uint8), mode="RGBA")


def numpy_alpha_composite(base, image):
    base = base.astype(np.float32)
    image = image.astype(np.float32)
    alpha_top = image[..., 3:4] / 255.0
    alpha_base = base[..., 3:4] / 255.0
    out_rgb = image[..., :3] * alpha_top + base[..., :3] * alpha_base * (1 - alpha_top)
    out_alpha = alpha_top + alpha_base * (1 - alpha_top)
    out = np.zeros_like(base)
    np.divide(out_rgb, out_alpha, out=out[..., :3], where=out_alpha > 0)
    out[..., 3] = out_alpha[..., 0] * 255
    # 直接 astype(uint8) 會截斷 29.999...，反覆合成時會逐片變暗。
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def apply_mask_alpha(fragment, mask):
    if mask is None:
        return fragment
    out = fragment.copy()
    out[..., 3] = np.minimum(out[..., 3], mask[..., 3])
    return out


def crop_to_primary_mask(image, primary_mask):
    if primary_mask is None:
        return image.copy()
    out = image.copy()
    primary_mask = np.asarray(primary_mask, dtype=bool)
    if primary_mask.shape != out.shape[:2]:
        raise ValueError(
            f"primary_mask shape {primary_mask.shape} doesn't match image shape {out.shape[:2]}"
        )
    out[~primary_mask] = 0
    return out


def secondary_overflow_split_parameters(block_size, random_factor):
    """雙遮罩外框使用較小區塊，限制溢進次要遮罩的深度。"""
    return (
        max(1, int(block_size) // 2),
        max(1, int(random_factor) // 2),
    )


def mask_coverage(mask, shape=None):
    """將 RGBA 或單通道遮罩轉為唯讀用途的布林覆蓋範圍。"""
    array = np.asarray(mask)
    if array.ndim == 2:
        coverage = array.astype(bool, copy=False)
    elif array.ndim == 3 and array.shape[2] >= 4:
        coverage = array[..., 3] > 0
    else:
        raise ValueError("遮罩必須是單通道或 RGBA 圖片")
    if shape is not None and coverage.shape != tuple(shape):
        raise ValueError(f"遮罩尺寸 {coverage.shape} 與圖片尺寸 {tuple(shape)} 不一致")
    return coverage


@dataclass(frozen=True)
class MaskWorkflowStages:
    """嚴格依工作順序保留的遮罩中間產物。"""

    primary_outer: np.ndarray
    primary_inner: np.ndarray
    secondary_inner: np.ndarray
    secondary_outer_source: np.ndarray
    secondary_outer_mask: np.ndarray | None

    def compose_final_base(self):
        """最後一步才把主體外框與次要內部合成清單第一張。"""
        return numpy_alpha_composite(self.primary_outer, self.secondary_inner)


def build_mask_workflow_stages(
    main_image,
    primary_mask=None,
    secondary_mask=None,
    *,
    primary_overflow=False,
    block_size=1,
    random_factor=1,
    seed=None,
):
    """按「主要內外 → 次要內外 → 外框分片」建立互不省略的階段資料。"""
    main = np.asarray(main_image, dtype=np.uint8)
    primary = (
        np.asarray(primary_mask, dtype=np.uint8)
        if primary_mask is not None
        else None
    )
    secondary = (
        np.asarray(secondary_mask, dtype=np.uint8)
        if secondary_mask is not None
        else None
    )

    if main.ndim != 3 or main.shape[2] < 4:
        raise ValueError("主圖必須是 RGBA 圖片")
    if primary is not None and primary.shape[:2] != main.shape[:2]:
        raise ValueError("主體遮罩尺寸必須與主圖一致")
    if secondary is not None and secondary.shape[:2] != main.shape[:2]:
        raise ValueError("次要遮罩尺寸必須與主圖一致")
    if primary is not None and (primary.ndim != 3 or primary.shape[2] < 4):
        raise ValueError("主體遮罩必須是 RGBA 圖片")
    if secondary is not None and (secondary.ndim != 3 or secondary.shape[2] < 4):
        raise ValueError("次要遮罩必須是 RGBA 圖片")

    main_visible = main[..., 3] > 0
    if primary is None and secondary is None:
        raise ValueError("遮罩流程至少需要一張遮罩")
    primary_area = (
        (primary[..., 3] > 0) & main_visible
        if primary is not None
        else main_visible.copy()
    )
    secondary_area = (
        (secondary[..., 3] > 0) & primary_area
        if secondary is not None
        else np.zeros(main.shape[:2], dtype=bool)
    )
    outer_area = (
        main_visible & ~primary_area
        if primary is not None
        else np.zeros(main.shape[:2], dtype=bool)
    )
    inner_frame_area = primary_area & ~secondary_area

    if primary is not None and not np.any(primary_area):
        raise ValueError("主體遮罩沒有覆蓋主圖的任何不透明像素")
    if secondary is not None and not np.any(secondary_area):
        raise ValueError("次要遮罩在主體遮罩內沒有任何有效範圍")
    if not np.any(inner_frame_area):
        raise ValueError("主體遮罩內、次要遮罩外沒有可拆分的像素")

    # 階段 1：主要內部永遠嚴格保留；只有外框可依開關向內部溢出。
    primary_outer = np.zeros_like(main)
    primary_outer[outer_area] = main[outer_area]
    if primary_overflow and np.any(outer_area):
        overflow_result = split_fragments(
            main,
            outer_area,
            1,
            block_size,
            random_factor,
            strict_mask=False,
            seed=seed,
        )
        if overflow_result:
            primary_outer = overflow_result[0]
    primary_inner = np.zeros_like(main)
    primary_inner[primary_area] = main[primary_area]

    # 階段 2：次要遮罩從「主要內部」切出內部與外框。
    secondary_inner = np.zeros_like(main)
    secondary_inner[secondary_area] = primary_inner[secondary_area]
    # 後續只讀取主要內部，不必再複製一張完整 RGBA。
    secondary_outer_source = primary_inner
    secondary_outer_mask = None
    if secondary is not None:
        secondary_outer_mask = inner_frame_area.copy()
    return MaskWorkflowStages(
        primary_outer=primary_outer,
        primary_inner=primary_inner,
        secondary_inner=secondary_inner,
        secondary_outer_source=secondary_outer_source,
        secondary_outer_mask=secondary_outer_mask,
    )


def build_mask_workflow_sources(main_image, primary_mask, secondary_mask=None):
    """相容舊呼叫；新流程應使用 build_mask_workflow_stages 保留階段。"""
    stages = build_mask_workflow_stages(main_image, primary_mask, secondary_mask)
    return (
        stages.compose_final_base(),
        stages.secondary_outer_source,
        stages.secondary_outer_mask,
    )


def ellipsis_middle(text, maxlen=28):
    if len(text) <= maxlen:
        return text
    name, ext = os.path.splitext(text)
    if len(ext) > 5:
        ext = ext[:5] + "..."
    keep = maxlen - len(ext) - 3
    if keep < 8:
        return text[:maxlen - 3] + "..."
    return name[:keep // 2] + "..." + name[-keep // 2:] + ext


def split_fragments(
    main_image,
    mask_image,
    count,
    block_size,
    random_factor,
    strict_mask=True,
    progress_cb=None,
    abort_cb=None,
    seed=None,
    rng=None,
    np_rng=None,
):
    """將 RGBA 圖片切成碎片；所有大量像素操作均使用 NumPy。"""
    if progress_cb:
        progress_cb(0, 7, "計算分割區塊...")

    rgba = np.asarray(main_image, dtype=np.uint8)
    rng = rng if rng is not None else random.Random(seed)
    np_rng = np_rng if np_rng is not None else np.random.default_rng(seed)
    h, w = rgba.shape[:2]
    if mask_image is not None:
        try:
            coverage_mask = mask_coverage(mask_image, (h, w))
        except ValueError:
            return []
    else:
        coverage_mask = rgba[..., 3] > 0

    if not np.any(coverage_mask):
        return []

    count = max(1, int(count))
    block_size = max(1, int(block_size))
    random_factor = max(1, int(random_factor))

    if progress_cb:
        progress_cb(1, 7, "開始分割...")
        progress_cb(2, 7, "正在產生分割區塊...")

    blocks = []
    y = 0
    while y < h:
        if abort_cb and abort_cb():
            return []
        block_h = rng.randint(block_size, block_size * random_factor)
        if y + block_h > h or h - y < block_size:
            block_h = h - y
        x = 0
        while x < w:
            block_w = rng.randint(block_size, block_size * random_factor)
            if x + block_w > w or w - x < block_size:
                block_w = w - x
            if np.any(coverage_mask[y:y + block_h, x:x + block_w]):
                blocks.append((x, y, block_w, block_h))
            x += block_w
        y += block_h

    if progress_cb:
        progress_cb(3, 7, "正在產生碎片（第1步）...")
    fragment_blocks = [[] for _ in range(count)]
    for block in blocks:
        fragment_blocks[rng.randrange(count)].append(block)

    if progress_cb:
        progress_cb(4, 7, "正在分配像素到碎片...")
    fragment_id_map = np.full((h, w), -1, dtype=np.int32)
    for fragment_id, assigned_blocks in enumerate(fragment_blocks):
        if abort_cb and abort_cb():
            return []
        for x, y, block_w, block_h in assigned_blocks:
            region = fragment_id_map[y:y + block_h, x:x + block_w]
            if strict_mask:
                region[coverage_mask[y:y + block_h, x:x + block_w]] = fragment_id
            else:
                region[:] = fragment_id

    if progress_cb:
        progress_cb(5, 7, "處理孤立像素...")
    unassigned = coverage_mask & (fragment_id_map == -1)
    unassigned_count = int(np.count_nonzero(unassigned))
    if unassigned_count:
        fragment_id_map[unassigned] = np_rng.integers(
            0, count, size=unassigned_count
        )

    if progress_cb:
        progress_cb(6, 7, "產生碎片圖像...")
    fragments = [np.zeros((h, w, 4), dtype=np.uint8) for _ in range(count)]
    for fragment_id, fragment in enumerate(fragments):
        assigned = fragment_id_map == fragment_id
        fragment[assigned] = rgba[assigned]
    return fragments


def split_fragments_with_secondary_overflow(
    main_image,
    secondary_outer_mask,
    count,
    block_size,
    random_factor,
    progress_cb=None,
    abort_cb=None,
    seed=None,
):
    """先建立次要外框的受控溢出，再以原始參數拆成最終碎片。

    溢出只發生在「次要內／外分離」階段。後續外框分片使用原始方塊
    尺寸與隨機度，並嚴格限制在前一階段得到的完整外框範圍內。
    """
    overflow_block_size, overflow_random_factor = (
        secondary_overflow_split_parameters(block_size, random_factor)
    )
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if progress_cb:
        progress_cb(0, 7, "建立次要外框溢出範圍...")
    overflow_result = split_fragments(
        main_image,
        secondary_outer_mask,
        1,
        overflow_block_size,
        overflow_random_factor,
        strict_mask=False,
        abort_cb=abort_cb,
        rng=rng,
        np_rng=np_rng,
    )
    if not overflow_result or (abort_cb and abort_cb()):
        return []
    return split_fragments(
        overflow_result[0],
        None,
        count,
        block_size,
        random_factor,
        strict_mask=True,
        progress_cb=progress_cb,
        abort_cb=abort_cb,
        rng=rng,
        np_rng=np_rng,
    )


def overlap_fill_fragment_worker(args):
    (
        fragment,
        main_image,
        mask_image,
        fill_percent,
        block_size,
        random_range,
        aggregation,
        limit_to_mask,
    ) = args[:8]
    abort_cb = args[8] if len(args) > 8 else None
    seed = args[9] if len(args) > 9 else None
    rng = np.random.default_rng(seed)

    if main_image is None or fill_percent <= 0:
        return fragment

    image = main_image
    h, w = image.shape[:2]
    if mask_image is not None and limit_to_mask:
        valid = mask_coverage(mask_image, (h, w)) & (image[..., 3] == 255)
    else:
        valid = image[..., 3] == 255

    fragment_valid = (fragment[..., 3] == 0) & valid
    coords = np.argwhere(fragment_valid)
    total = int(coords.shape[0])
    if total == 0:
        return fragment

    target_fill = int(total * float(fill_percent) / 100.0)
    if target_fill <= 0:
        return fragment

    filled = np.zeros((h, w), dtype=bool)
    out = fragment.copy()
    aggregation_ratio = min(max(int(aggregation), 1), 10) / 10.0
    cluster_centers = []
    fill_count = 0
    tries = 0
    fail_streak = 0
    max_tries = max(1000, target_fill * 8)

    while fill_count < target_fill and tries < max_tries and fail_streak < 4000:
        if abort_cb and abort_cb():
            return fragment
        if cluster_centers and rng.random() < aggregation_ratio:
            base_y, base_x = cluster_centers[int(rng.integers(len(cluster_centers)))]
            y = base_y + int(rng.integers(-block_size, block_size + 1))
            x = base_x + int(rng.integers(-block_size, block_size + 1))
            if not (0 <= y < h and 0 <= x < w and fragment_valid[y, x]):
                y, x = coords[int(rng.integers(len(coords)))]
        else:
            y, x = coords[int(rng.integers(len(coords)))]

        y, x = int(y), int(x)
        max_size = min(block_size * max(1, int(random_range)), h - y, w - x)
        if y < 0 or x < 0 or max_size < block_size:
            tries += 1
            fail_streak += 1
            continue
        size = int(rng.integers(block_size, max_size + 1))
        patch = image[y:y + size, x:x + size]
        target = (
            ~filled[y:y + size, x:x + size]
            & (patch[..., 3] == 255)
            & fragment_valid[y:y + size, x:x + size]
        )
        if not np.any(target):
            tries += 1
            fail_streak += 1
            continue

        out[y:y + size, x:x + size][target] = patch[target]
        filled[y:y + size, x:x + size][target] = True
        increment = int(np.count_nonzero(target))
        fill_count += increment
        fail_streak = 0
        ys, xs = np.where(target)
        if ys.size:
            for index in rng.choice(ys.size, size=min(5, ys.size), replace=False):
                cluster_centers.append((y + int(ys[index]), x + int(xs[index])))
        tries += 1
    return out


def apply_overlap_to_all_fragments(
    fragments,
    main_image,
    mask_image,
    fill_percent,
    block_size,
    random_range,
    progress_cb=None,
    abort_cb=None,
    limit_to_mask=True,
    aggregation=1,
    seed=None,
):
    """使用共用記憶體的執行緒池，避免為每片重複複製主圖與遮罩。"""
    count = len(fragments)
    if fill_percent <= 0 or count == 0 or main_image is None:
        return fragments

    child_sequences = np.random.SeedSequence(seed).spawn(count)
    args_list = [
        (
            fragment,
            main_image,
            mask_image,
            fill_percent,
            block_size,
            random_range,
            aggregation,
            limit_to_mask,
            abort_cb,
            int(child_sequences[index].generate_state(1, dtype=np.uint64)[0]),
        )
        for index, fragment in enumerate(fragments)
    ]
    results = [None] * count
    with ThreadPoolExecutor(max_workers=min(4, count)) as executor:
        futures = {
            executor.submit(overlap_fill_fragment_worker, args): index
            for index, args in enumerate(args_list)
        }
        completed = 0
        for future in as_completed(futures):
            if abort_cb and abort_cb():
                for pending in futures:
                    pending.cancel()
                return []
            results[futures[future]] = future.result()
            completed += 1
            if progress_cb:
                progress_cb(completed, count, "重疊像素合成...")
    return results
