"""Data loading and episodic texture-transfer dataset utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spritecraft.config import (
    DATASET_PATH,
    PAIR_INDEX_PATH,
    VALIDATION_FILENAMES,
)


class TextureDataset(Dataset):
    """PyTorch-compatible dataset for pack-conditioned texture transfer episodes."""

    def __init__(
        self,
        pair_index_path: Path = PAIR_INDEX_PATH,
        dataset_path: Path = DATASET_PATH,
        split: str = "train",
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        self.split = split

        with np.load(dataset_path) as dataset_file:
            self.data = {pack_id: dataset_file[pack_id] for pack_id in dataset_file.files}

        with open(pair_index_path, encoding="utf-8") as file_obj:
            pair_data = json.load(file_obj)

        split_pairs = pair_data.get(split)
        if split_pairs is None:
            raise KeyError(f"Split {split!r} not found in {pair_index_path}")

        self.pack_ids = pair_data.get("pack_ids", [])
        self.base_pack_idx = pair_data.get("base_pack_idx")
        if self.base_pack_idx is None:
            raise KeyError(f"base_pack_idx not found in {pair_index_path}")
        if self.base_pack_idx >= len(self.pack_ids):
            raise ValueError(f"base_pack_idx {self.base_pack_idx} out of range for {len(self.pack_ids)} packs")

        self.base_pack_id = self.pack_ids[self.base_pack_idx]
        if self.base_pack_id not in self.data:
            raise KeyError(f"Base pack {self.base_pack_id!r} not found in {dataset_path}")

        self.pack_roles = pair_data.get("pack_roles", {})
        self.pack_styles = pair_data.get("pack_styles", {})
        self.validation_filenames = set(pair_data.get("validation_filenames", sorted(VALIDATION_FILENAMES)))
        self.validation_matrix = [
            entry for entry in pair_data.get("validation_matrix", []) if entry.get("split") == split
        ]
        self.filename_to_index_per_pack = {
            pack_id: {filename: idx for idx, filename in enumerate(filenames)}
            for pack_id, filenames in pair_data.get("filenames_per_pack", {}).items()
        }
        self.episodes = self._build_episodes(split_pairs)
        self.episode_lookup = {
            (episode["filename"], episode["target_pack"]): idx
            for idx, episode in enumerate(self.episodes)
        }
        self.filenames = sorted({episode["filename"] for episode in self.episodes})

    def _build_episodes(self, split_pairs: dict[str, list[list[str | int]]]) -> list[dict[str, str | int]]:
        base_lookup = self.filename_to_index_per_pack.get(self.base_pack_id, {})
        episodes: list[dict[str, str | int]] = []

        for filename in sorted(split_pairs):
            if filename not in base_lookup:
                continue

            for pair in split_pairs[filename]:
                if len(pair) < 2:
                    continue
                pack_idx = pair[0]
                if not isinstance(pack_idx, int):
                    continue
                if pack_idx == self.base_pack_idx:
                    continue
                if pack_idx < 0 or pack_idx >= len(self.pack_ids):
                    continue
                target_pack = self.pack_ids[pack_idx]
                if target_pack not in self.filename_to_index_per_pack:
                    continue

                style = str(self.pack_styles.get(target_pack, "unspecified"))
                episodes.append(
                    {
                        "filename": filename,
                        "target_pack": target_pack,
                        "target_pack_idx": pack_idx,
                        "style": style,
                    }
                )

        if not episodes:
            raise ValueError(
                "No transfer episodes could be constructed. Re-run preprocessing and ensure the base pack "
                "shares non-validation textures with at least one target pack."
            )

        return episodes

    def __len__(self):
        return len(self.episodes)

    def _lookup_texture(self, pack_id: str, filename: str) -> torch.Tensor:
        array_idx = self.filename_to_index_per_pack[pack_id][filename]
        return torch.as_tensor(self.data[pack_id][array_idx], dtype=torch.long)

    def get_episode_index(self, filename: str, target_pack: str) -> int:
        try:
            return self.episode_lookup[(filename, target_pack)]
        except KeyError as exc:
            raise ValueError(f"Episode ({filename!r}, {target_pack!r}) is not present in split {self.split!r}") from exc

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        episode = self.episodes[idx]
        filename = episode["filename"]
        target_pack = episode["target_pack"]
        target_pack_idx = episode["target_pack_idx"]

        source = self._lookup_texture(self.base_pack_id, filename)
        target = self._lookup_texture(target_pack, filename)

        return {
            "filename": filename,
            "source": source,
            "pack_id": torch.tensor(target_pack_idx),
            "target": target,
        }
