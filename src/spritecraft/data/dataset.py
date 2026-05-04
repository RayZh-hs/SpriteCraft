"""Data loading and dataset utilities."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spritecraft.config import PAIR_INDEX_PATH, DATASET_PATH


class TextureDataset(Dataset):
    """PyTorch-compatible dataset for paired texture blocks."""

    def __init__(
        self,
        pair_index_path: Path = PAIR_INDEX_PATH,
        dataset_path: Path = DATASET_PATH,
        split: str = "train",
        seed: int = 42,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        self.split = split
        self._rng = random.Random(seed)

        with np.load(dataset_path) as dataset_file:
            self.data = {pack_id: dataset_file[pack_id] for pack_id in dataset_file.files}

        with open(pair_index_path) as f:
            pair_data = json.load(f)

        split_pairs = pair_data.get(split)
        if split_pairs is None:
            raise KeyError(f"Split {split!r} not found in {pair_index_path}")

        self.filenames_per_pack = pair_data.get("filenames_per_pack", {})
        self.filenames = sorted(split_pairs)
        self.pair_index = {
            filename: [(pack_id, int(array_idx)) for pack_id, array_idx in split_pairs[filename]]
            for filename in self.filenames
        }

    def __len__(self):
        return len(self.filenames)

    def _deterministic_rng(self, filename: str, idx: int) -> random.Random:
        seed = sum((char_idx + 1) * ord(char) for char_idx, char in enumerate(filename))
        return random.Random(seed + idx)

    @staticmethod
    def _pick_style_index(
        style_source: np.ndarray,
        target_idx: int,
        rng: random.Random,
    ) -> int:
        if len(style_source) <= 1:
            return target_idx

        sampled_idx = rng.randrange(len(style_source) - 1)
        if sampled_idx >= target_idx:
            sampled_idx += 1
        return sampled_idx

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        filename = self.filenames[idx]
        pairs = self.pair_index[filename]
        if len(pairs) < 2:
            raise ValueError(f"Need at least two pack variants for {filename}, found {len(pairs)}")

        rng = self._rng if self.split == "train" else self._deterministic_rng(filename, idx)
        content_pair, target_pair = rng.sample(pairs, k=2)
        content_pack, content_idx = content_pair
        target_pack, target_idx = target_pair

        target_pack_array = self.data[target_pack]
        style_idx = self._pick_style_index(target_pack_array, target_idx, rng)

        content_ref = torch.as_tensor(self.data[content_pack][content_idx], dtype=torch.long)
        style_ref = torch.as_tensor(target_pack_array[style_idx], dtype=torch.long)
        target = torch.as_tensor(target_pack_array[target_idx], dtype=torch.long)

        return {
            "filename": filename,
            "content_filename": filename,
            "style_filename": self.filenames_per_pack[target_pack][style_idx],
            "target_filename": filename,
            "content_pack": content_pack,
            "style_pack": target_pack,
            "target_pack": target_pack,
            "content_ref": content_ref,
            "style_ref": style_ref,
            "target": target,
        }
