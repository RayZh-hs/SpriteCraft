"""Data loading and dataset utilities."""

import json
import numpy as np
from pathlib import Path

from spritecraft.config import PAIR_INDEX_PATH, DATASET_PATH


class TextureDataset:
    """PyTorch-compatible dataset for paired texture blocks."""

    def __init__(self, pair_index_path: Path = PAIR_INDEX_PATH, dataset_path: Path = DATASET_PATH, split: str = "train"):
        self.split = split
        self.data = np.load(dataset_path)
        with open(pair_index_path) as f:
            self.pair_index = json.load(f)

    def __len__(self):
        return len(self.pair_index)

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset indexing not yet implemented")
