"""Export utilities: palette mapping and resource-pack formatting."""

import numpy as np
from PIL import Image
from pathlib import Path

from spritecraft.config import PALETTE_PATH, IMAGE_SIZE


def indices_to_image(indices: np.ndarray, palette_path: Path = PALETTE_PATH, target_size: int = IMAGE_SIZE) -> Image.Image:
    """Convert palette indices back to RGB image."""
    palette = np.load(palette_path)
    rgb = palette[indices]
    img = Image.fromarray(rgb.astype(np.uint8))
    if target_size != IMAGE_SIZE:
        img = img.resize((target_size, target_size), Image.Resampling.NEAREST)
    return img
