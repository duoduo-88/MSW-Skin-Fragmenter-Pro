import numpy as np

from .fragmentation import numpy_alpha_composite


class _MaskCoordinateSampler:
    """密集遮罩用拒絕抽樣；稀疏遮罩只保存一維索引。"""

    def __init__(self, mask, sparse_threshold=0.20):
        self.mask = np.asarray(mask, dtype=bool)
        self.height, self.width = self.mask.shape
        self.valid_count = int(np.count_nonzero(self.mask))
        self.flat_indices = None
        if self.valid_count == 0:
            self.bounds = None
            return

        rows = np.flatnonzero(np.any(self.mask, axis=1))
        cols = np.flatnonzero(np.any(self.mask, axis=0))
        self.bounds = (
            int(rows[0]),
            int(rows[-1]) + 1,
            int(cols[0]),
            int(cols[-1]) + 1,
        )
        y0, y1, x0, x1 = self.bounds
        bounding_area = max(1, (y1 - y0) * (x1 - x0))
        if self.valid_count / bounding_area < sparse_threshold:
            self.flat_indices = np.flatnonzero(self.mask.ravel())

    def sample(self, rng):
        if self.valid_count == 0:
            return None
        if self.flat_indices is not None:
            flat = int(self.flat_indices[int(rng.integers(len(self.flat_indices)))])
            return divmod(flat, self.width)

        y0, y1, x0, x1 = self.bounds
        for _ in range(32):
            y = int(rng.integers(y0, y1))
            x = int(rng.integers(x0, x1))
            if self.mask[y, x]:
                return y, x

        # 極端不規則遮罩才延遲建立一維索引，避免無限拒絕抽樣。
        self.flat_indices = np.flatnonzero(self.mask.ravel())
        if not len(self.flat_indices):
            return None
        flat = int(self.flat_indices[int(rng.integers(len(self.flat_indices)))])
        return divmod(flat, self.width)


def build_interfere_block_pool(
    sources,
    block_size=10,
    random_range=2,
    pool_size=400,
    alpha_min=1,
    alpha_max=100,
    restrict_mask=None,
    max_pool_bytes=32 * 1024 * 1024,
    rng=None,
    abort_cb=None,
):
    """從一張或多張 RGBA 圖建立有記憶體上限的干擾素材池。"""
    if isinstance(sources, np.ndarray):
        sources = [sources]
    else:
        sources = list(sources or [])

    block_size = max(1, int(block_size))
    random_range = max(1, int(random_range))
    pool_size = max(0, int(pool_size))
    alpha_min, alpha_max = sorted((float(alpha_min), float(alpha_max)))
    rng = rng if rng is not None else np.random.default_rng()

    candidates = []
    for source in sources:
        if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] < 4:
            continue
        h, w = source.shape[:2]
        if h < block_size or w < block_size:
            continue
        valid = source[..., 3] > 0
        if restrict_mask is not None:
            mask = np.asarray(restrict_mask, dtype=bool)
            if mask.shape != (h, w):
                continue
            valid &= mask
        eligible = valid.copy()
        if block_size > 1:
            eligible[h - block_size + 1:, :] = False
            eligible[:, w - block_size + 1:] = False
        sampler = _MaskCoordinateSampler(eligible)
        if sampler.valid_count:
            candidates.append((source, sampler))

    if not candidates or pool_size == 0:
        return []

    blocks = []
    pool_bytes = 0
    max_attempts = max(pool_size * 20, 200)
    for _ in range(max_attempts):
        if abort_cb and abort_cb():
            return []
        source, sampler = candidates[int(rng.integers(len(candidates)))]
        h, w = source.shape[:2]
        point = sampler.sample(rng)
        if point is None:
            continue
        y, x = point
        max_size = min(block_size * random_range, h - y, w - x)
        if max_size < block_size:
            continue
        size = int(rng.integers(block_size, max_size + 1))
        patch = source[y:y + size, x:x + size]
        alpha_area = np.count_nonzero(patch[..., 3]) / patch[..., 3].size * 100
        if not alpha_min <= alpha_area <= alpha_max:
            continue
        if patch.nbytes > max_pool_bytes or pool_bytes + patch.nbytes > max_pool_bytes:
            continue
        blocks.append(patch.copy())
        pool_bytes += patch.nbytes
        if len(blocks) >= pool_size:
            break
    return blocks


def build_random_previous_scope(previous_fragments, ignore_semitrans=True, rng=None):
    """隨機選一片或多片前方碎片，回傳它們 alpha 範圍的聯集。"""
    candidates = [
        (name, fragment)
        for name, fragment in previous_fragments
        if isinstance(fragment, np.ndarray) and fragment.ndim == 3 and fragment.shape[2] >= 4
    ]
    if not candidates:
        return None, []

    rng = rng if rng is not None else np.random.default_rng()

    shape = candidates[0][1].shape[:2]
    candidates = [item for item in candidates if item[1].shape[:2] == shape]
    if not candidates:
        return None, []

    selected_count = int(rng.integers(1, len(candidates) + 1))
    selected_indices = rng.choice(
        len(candidates), size=selected_count, replace=False
    )
    selected = [candidates[int(index)] for index in np.atleast_1d(selected_indices)]
    scope = np.zeros(shape, dtype=bool)
    for _, fragment in selected:
        if ignore_semitrans:
            scope |= fragment[..., 3] == 255
        else:
            scope |= fragment[..., 3] > 0
    return scope, [name for name, _ in selected]


def gen_multi_overlap_interfere(
    fragment,
    block_pool,
    coverage=0.7,
    max_try=10,
    allow_overlap=False,
    primary_mask=None,
    rng=None,
    abort_cb=None,
):
    h, w = fragment.shape[:2]
    valid_mask = (
        np.asarray(primary_mask, dtype=bool)
        if primary_mask is not None
        else fragment[..., 3] > 0
    )
    if valid_mask.shape != (h, w):
        raise ValueError("干擾遮罩與碎片尺寸不一致")
    rng = rng if rng is not None else np.random.default_rng()
    sampler = _MaskCoordinateSampler(valid_mask)
    total_valid = sampler.valid_count
    if total_valid == 0 or not block_pool:
        return np.zeros_like(fragment)

    out = np.zeros_like(fragment)
    pasted_mask = np.zeros((h, w), dtype=bool)
    pasted = 0
    tries = 0
    coverage = float(np.clip(coverage, 0.0, 1.0))
    target_fill = int(total_valid * coverage)
    if target_fill <= 0:
        return out
    max_tries = max(1000, target_fill * max(1, int(max_try)))

    while pasted < target_fill and tries < max_tries:
        if abort_cb and abort_cb():
            return np.zeros_like(fragment)
        point = sampler.sample(rng)
        if point is None:
            break
        y, x = point
        patch = block_pool[int(rng.integers(len(block_pool)))]
        block_h, block_w = patch.shape[:2]
        if y + block_h > h or x + block_w > w:
            tries += 1
            continue

        local_pasted = pasted_mask[y:y + block_h, x:x + block_w]
        valid_patch = (patch[..., 3] > 0) & valid_mask[y:y + block_h, x:x + block_w]
        if not allow_overlap:
            valid_patch &= ~local_pasted
        if not np.any(valid_patch):
            tries += 1
            continue

        out_region = out[y:y + block_h, x:x + block_w]
        out_region[valid_patch] = numpy_alpha_composite(
            out_region[valid_patch], patch[valid_patch]
        )
        newly_pasted = valid_patch & ~local_pasted
        pasted += int(np.count_nonzero(newly_pasted))
        local_pasted |= valid_patch
        tries += 1
    return out


def generate_interference_worker(args):
    fragment_name, fragment, scope_candidates, base_pool, settings = args[:5]
    seed = args[5] if len(args) > 5 else None
    abort_event = args[6] if len(args) > 6 else None
    rng = np.random.default_rng(seed)
    abort_cb = abort_event.is_set if abort_event is not None else None

    scope_mask, selected_names = build_random_previous_scope(
        scope_candidates,
        ignore_semitrans=settings.get("ignore_semitrans", True),
        rng=rng,
    )
    if scope_mask is None:
        return fragment_name, np.zeros_like(fragment), []

    generation_limit_mask = settings.get("generation_limit_mask")
    if generation_limit_mask is not None:
        generation_limit_mask = np.asarray(generation_limit_mask, dtype=bool)
        if generation_limit_mask.shape != scope_mask.shape:
            raise ValueError("局部干擾範圍與碎片尺寸不一致")
        scope_mask &= generation_limit_mask

    interference = gen_multi_overlap_interfere(
        fragment,
        base_pool,
        coverage=settings["density"],
        allow_overlap=settings.get("allow_overlap", False),
        primary_mask=scope_mask,
        rng=rng,
        abort_cb=abort_cb,
    )
    return fragment_name, interference, selected_names


def apply_interference_masked(original, interference, mask=None):
    out = original.copy()
    valid = interference[..., 3] > 0
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    out[valid] = numpy_alpha_composite(original[valid], interference[valid])
    return out
