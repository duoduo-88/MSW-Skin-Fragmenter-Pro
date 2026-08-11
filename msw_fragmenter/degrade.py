import random
import secrets

import numpy as np


def simple_block_degrade(
    source,
    block_size,
    random_range,
    density,
    noise_strength,
    brightness_strength,
    color_strength,
):
    """只改變可見像素的 RGB，完整保留原始 alpha。"""
    h, w = source.shape[:2]
    out = source.copy().astype(np.float32)
    alpha_mask = source[..., 3] > 0
    valid_coords = np.argwhere(alpha_mask)
    valid_count = int(valid_coords.shape[0])
    if valid_count == 0 or density <= 0:
        out[..., 3] = source[..., 3]
        return np.clip(out, 0, 255).astype(np.uint8)

    block_size = max(1, int(block_size))
    random_range = max(1, int(random_range))
    block_count = max(1, int(valid_count * float(density) / (block_size * block_size)))

    for _ in range(block_count):
        size = random.randint(block_size, block_size * random_range)
        y, x = valid_coords[random.randrange(valid_count)]
        y = int(min(max(0, y - size // 2), max(0, h - size)))
        x = int(min(max(0, x - size // 2), max(0, w - size)))
        region = out[y:y + size, x:x + size, :3]
        if not region.size:
            continue

        brightness = 1 + random.uniform(-brightness_strength, brightness_strength) / 100.0
        patch = region * brightness
        channel_offsets = np.array(
            [random.uniform(-color_strength, color_strength) for _ in range(3)],
            dtype=np.float32,
        ) / 100.0
        patch += patch * channel_offsets
        sigma = noise_strength / 100.0 * 30
        if sigma > 0:
            patch += np.random.normal(0, sigma, patch.shape)
        patch = np.clip(patch, 0, 255)

        local_mask = alpha_mask[y:y + size, x:x + size]
        if np.any(local_mask):
            region[local_mask] = patch[local_mask]

    out[..., 3] = source[..., 3]
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_chunk_worker(args):
    chunk, settings = args
    return simple_block_degrade(
        chunk,
        settings["block_size"],
        settings["rand_range"],
        settings["density"],
        settings["noise_strength"],
        settings["brightness_strength"],
        settings["color_strength"],
    )


def apply_subtle_fragment_artifacts(
    fragments,
    reference_fragments=None,
    gap_ratio=0.16,
    shift_ratio=0.04,
    seed=None,
    progress_cb=None,
    abort_cb=None,
):
    """Add short 1 px gap and offset segments along inter-fragment outlines.

    The same artifact plan is applied to the visible fragments and the
    interference-free snapshot so that restoring the initial split preserves
    the intended degraded result.
    """
    if not fragments:
        return [], []
    references = reference_fragments or fragments
    if len(references) != len(fragments):
        raise ValueError("碎片與接縫參考數量不一致")
    shape = fragments[0].shape
    if len(shape) != 3 or shape[2] != 4:
        raise ValueError("劣化碎片必須是 RGBA 圖像")
    if any(image.shape != shape for image in fragments + references):
        raise ValueError("劣化碎片尺寸不一致")

    h, w = shape[:2]
    rng = np.random.default_rng(seed if seed is not None else secrets.randbits(64))
    # Shift every fragment independently, then detect its new outline. Each
    # fragment remains on its own layer, so natural 1 px overlaps are kept.
    # Vacated pixels are extended from the fragment's original edge, keeping
    # the random offset without opening an extra transparent seam.
    fragment_directions = np.array(
        ((-1, 0), (1, 0), (0, -1), (0, 1)), dtype=np.int16
    )
    offsets = fragment_directions[
        rng.integers(0, len(fragment_directions), size=len(fragments))
    ]

    def shift_whole_fragments(images):
        shifted_images = []
        for image, (delta_y, delta_x) in zip(images, offsets):
            delta_y, delta_x = int(delta_y), int(delta_x)
            shifted = np.zeros_like(image)
            source_y0, source_y1 = max(0, -delta_y), h - max(0, delta_y)
            source_x0, source_x1 = max(0, -delta_x), w - max(0, delta_x)
            target_y0, target_y1 = max(0, delta_y), h - max(0, -delta_y)
            target_x0, target_x1 = max(0, delta_x), w - max(0, -delta_x)
            shifted[target_y0:target_y1, target_x0:target_x1] = image[
                source_y0:source_y1, source_x0:source_x1
            ]
            vacated = (image[..., 3] > 0) & (shifted[..., 3] == 0)
            shifted[vacated] = image[vacated]
            shifted_images.append(shifted)
        return shifted_images

    fragments = shift_whole_fragments(fragments)
    references = shift_whole_fragments(references)
    owner = np.full((h, w), -1, dtype=np.int16)
    best_alpha = np.zeros((h, w), dtype=np.uint8)
    total_steps = max(1, len(references) + 4)
    for index, image in enumerate(references):
        if abort_cb and abort_cb():
            return [], []
        alpha = image[..., 3]
        replace = alpha > best_alpha
        owner[replace] = index
        best_alpha[replace] = alpha[replace]
        if progress_cb:
            progress_cb(index + 1, total_steps, "分析碎片拼接邊界...")

    boundary = np.zeros((h, w), dtype=bool)
    if w > 1:
        horizontal = (
            (owner[:, :-1] >= 0)
            & (owner[:, 1:] >= 0)
            & (owner[:, :-1] != owner[:, 1:])
        )
        boundary[:, 1:] |= horizontal
    if h > 1:
        vertical = (
            (owner[:-1, :] >= 0)
            & (owner[1:, :] >= 0)
            & (owner[:-1, :] != owner[1:, :])
        )
        boundary[1:, :] |= vertical
    coordinates = np.argwhere(boundary)
    if coordinates.size == 0:
        return [image.copy() for image in fragments], [
            image.copy() for image in references
        ]

    count = len(coordinates)
    gap_count = min(count, max(0, int(round(count * float(gap_ratio)))))
    shift_count = min(
        count - gap_count,
        max(0, int(round(count * float(shift_ratio)))) if count > gap_count else 0,
    )
    def sample_segments(target_count, min_length, max_length, blocked=None):
        selected_mask = np.zeros((h, w), dtype=bool)
        if blocked is not None:
            selected_mask |= blocked
        segments = []
        selected_count = 0
        attempts = 0
        max_attempts = max(64, target_count * 8)
        neighbor_steps = np.array(
            ((-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1)),
            dtype=np.int16,
        )
        while selected_count < target_count and attempts < max_attempts:
            attempts += 1
            start_y, start_x = coordinates[rng.integers(0, count)]
            if selected_mask[start_y, start_x]:
                continue
            desired = int(rng.integers(min_length, max_length + 1))
            segment = []
            y, x = int(start_y), int(start_x)
            previous_step = None
            while len(segment) < desired:
                if selected_mask[y, x] or not boundary[y, x]:
                    break
                selected_mask[y, x] = True
                segment.append((y, x))
                candidates = []
                for step_y, step_x in neighbor_steps:
                    next_y, next_x = y + int(step_y), x + int(step_x)
                    if (
                        0 <= next_y < h
                        and 0 <= next_x < w
                        and boundary[next_y, next_x]
                        and not selected_mask[next_y, next_x]
                    ):
                        score = rng.random() * 0.2
                        if previous_step is not None:
                            score += (
                                int(step_y) * previous_step[0]
                                + int(step_x) * previous_step[1]
                            )
                        candidates.append((score, next_y, next_x, int(step_y), int(step_x)))
                if not candidates:
                    break
                _, next_y, next_x, step_y, step_x = max(candidates, key=lambda item: item[0])
                y, x = next_y, next_x
                previous_step = (step_y, step_x)
            if segment:
                segments.append(segment)
                selected_count += len(segment)
        return segments

    gap_segments = sample_segments(gap_count, 4, 14)
    gap_coords = np.array(
        [point for segment in gap_segments for point in segment], dtype=np.int64
    )
    blocked = np.zeros((h, w), dtype=bool)
    if gap_coords.size:
        blocked[gap_coords[:, 0], gap_coords[:, 1]] = True
    shift_segments = sample_segments(shift_count, 3, 11, blocked=blocked)
    shift_coords = np.array(
        [point for segment in shift_segments for point in segment], dtype=np.int64
    )
    shift_segment_ids = np.array(
        [index for index, segment in enumerate(shift_segments) for _ in segment],
        dtype=np.int64,
    )
    gap_count = len(gap_coords)
    shift_count = len(shift_coords)
    directions = np.array(((-1, 0), (1, 0), (0, -1), (0, 1)), dtype=np.int16)
    segment_directions = directions[
        rng.integers(0, len(directions), size=len(shift_segments))
    ]
    shift_dirs = (
        segment_directions[shift_segment_ids]
        if shift_count
        else np.empty((0, 2), dtype=np.int16)
    )

    def apply_plan(images, message, step):
        output = [image.copy() for image in images]

        def restore_sixty_percent_alpha(
            y, x, source_owners, source_pixels
        ):
            for fragment_index, image in enumerate(output):
                choose = source_owners == fragment_index
                if not np.any(choose):
                    continue
                pixels = source_pixels[fragment_index][choose].copy()
                alpha = pixels[:, 3].astype(np.uint16)
                pixels[:, 3] = ((alpha * 3 + 2) // 5).astype(np.uint8)
                image[y[choose], x[choose]] = pixels

        if gap_count:
            gy, gx = gap_coords[:, 0], gap_coords[:, 1]
            gap_owners = owner[gy, gx]
            gap_source_pixels = [image[gy, gx].copy() for image in output]
            for image in output:
                image[gy, gx] = 0
            restore_sixty_percent_alpha(
                gy, gx, gap_owners, gap_source_pixels
            )
        if shift_count:
            sy, sx = shift_coords[:, 0], shift_coords[:, 1]
            source_owners = owner[sy, sx]
            source_pixels = [image[sy, sx].copy() for image in images]
            # Keep a 60%-alpha copy at the source seam, then put the owning
            # full-alpha pixel one pixel away.
            for image in output:
                image[sy, sx] = 0
            restore_sixty_percent_alpha(
                sy, sx, source_owners, source_pixels
            )
            dy = sy + shift_dirs[:, 0]
            dx = sx + shift_dirs[:, 1]
            valid = (dy >= 0) & (dy < h) & (dx >= 0) & (dx < w)
            for fragment_index, image in enumerate(output):
                choose = valid & (source_owners == fragment_index)
                if np.any(choose):
                    image[dy[choose], dx[choose]] = source_pixels[fragment_index][choose]
        if progress_cb:
            progress_cb(step, total_steps, message)
        return output

    processed = apply_plan(fragments, "加入輪廓錯位線段...", total_steps - 1)
    processed_references = apply_plan(
        references, "完成輪廓缺口線段...", total_steps
    )
    return processed, processed_references
