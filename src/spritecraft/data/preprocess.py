"""Preprocessing pipeline."""

from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from spritecraft.config import RAW_PACKS_DIR, PROCESSED_DIR, PALETTE_PATH


def run(packs_dir: str | Path = RAW_PACKS_DIR):
    """Run full preprocessing pipeline."""
    packs_dir = Path(packs_dir)
    # TODO: Implement filtering, resizing, palette extraction, quantization, pair indexing
    raise NotImplementedError("Preprocessing pipeline not yet implemented")
