"""Image preprocessing for EmbeddingGemma 2, ported from transformers' Gemma4ImageProcessorPil."""

import math

import numpy as np
from PIL import Image, ImageOps

PATCH_SIZE = 16
POOLING_KERNEL = 3
SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def target_size(height: int, width: int, max_patches: int) -> tuple[int, int]:
    """Largest aspect-preserving size within `max_patches` with sides divisible by 48px."""
    target_px = max_patches * PATCH_SIZE**2
    factor = math.sqrt(target_px / (height * width))
    side_mult = POOLING_KERNEL * PATCH_SIZE
    target_h = int(math.floor(factor * height / side_mult)) * side_mult
    target_w = int(math.floor(factor * width / side_mult)) * side_mult

    max_side = (max_patches // POOLING_KERNEL**2) * side_mult
    if target_h == 0:
        target_h = side_mult
        target_w = min(int(math.floor(width / height)) * side_mult, max_side)
    elif target_w == 0:
        target_w = side_mult
        target_h = min(int(math.floor(height / width)) * side_mult, max_side)
    return target_h, target_w


def preprocess_image(image: Image.Image, soft_tokens: int = 70) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (patches [max_patches, 768], positions [max_patches, 2], num_soft_tokens).

    Patches are padded to a fixed `max_patches` so the vision encoder always sees one shape.
    """
    if soft_tokens not in SUPPORTED_SOFT_TOKENS:
        raise ValueError(f"soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}")
    max_patches = soft_tokens * POOLING_KERNEL**2

    image = ImageOps.exif_transpose(image).convert("RGB")
    h, w = target_size(image.height, image.width, max_patches)
    if (h, w) != (image.height, image.width):
        image = image.resize((w, h), Image.Resampling.BICUBIC)

    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (C, H, W)
    ph, pw = h // PATCH_SIZE, w // PATCH_SIZE
    patches = (
        pixels.reshape(3, ph, PATCH_SIZE, pw, PATCH_SIZE)
        .transpose(1, 3, 2, 4, 0)
        .reshape(ph * pw, -1)
    )
    grid_x, grid_y = np.meshgrid(np.arange(pw), np.arange(ph), indexing="xy")
    positions = np.stack([grid_x, grid_y], axis=-1).reshape(ph * pw, 2)

    pad = max_patches - patches.shape[0]
    patches = np.pad(patches, [(0, pad), (0, 0)])
    positions = np.pad(positions, [(0, pad), (0, 0)], constant_values=-1)
    return patches, positions.astype(np.int64), (ph * pw) // POOLING_KERNEL**2
