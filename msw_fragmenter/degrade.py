import random

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
