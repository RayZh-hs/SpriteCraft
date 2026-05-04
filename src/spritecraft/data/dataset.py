"""Data loading and episodic texture-transfer dataset utilities."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spritecraft.config import (
    DATASET_PATH,
    NUM_SUPPORT_EXEMPLARS,
    PAIR_INDEX_PATH,
    VALIDATION_FILENAMES,
)


class TextureDataset(Dataset):
    """PyTorch-compatible dataset for vanilla-anchored texture transfer episodes."""

    def __init__(
        self,
        pair_index_path: Path = PAIR_INDEX_PATH,
        dataset_path: Path = DATASET_PATH,
        split: str = "train",
        seed: int = 42,
        num_support_exemplars: int = NUM_SUPPORT_EXEMPLARS,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        if num_support_exemplars <= 0:
            raise ValueError("num_support_exemplars must be positive")

        self.split = split
        self.num_support_exemplars = num_support_exemplars
        self._rng = random.Random(seed)

        with np.load(dataset_path) as dataset_file:
            self.data = {pack_id: dataset_file[pack_id] for pack_id in dataset_file.files}

        with open(pair_index_path, encoding="utf-8") as file_obj:
            pair_data = json.load(file_obj)

        split_pairs = pair_data.get(split)
        if split_pairs is None:
            raise KeyError(f"Split {split!r} not found in {pair_index_path}")

        self.filenames_per_pack = pair_data.get("filenames_per_pack", {})
        self.base_pack_id = pair_data.get("base_pack_id")
        if self.base_pack_id is None:
            self.base_pack_id = self._detect_base_pack_id(sorted(self.data))
        if self.base_pack_id not in self.data:
            raise KeyError(f"Base pack {self.base_pack_id!r} not found in {dataset_path}")

        self.validation_filenames = set(pair_data.get("validation_filenames", sorted(VALIDATION_FILENAMES)))
        self.filename_to_index_per_pack = {
            pack_id: {filename: idx for idx, filename in enumerate(filenames)}
            for pack_id, filenames in self.filenames_per_pack.items()
        }
        self.support_filenames_by_pack = self._build_support_filenames()
        self.episodes = self._build_episodes(split_pairs)
        self.filenames = sorted({episode["filename"] for episode in self.episodes})

    @staticmethod
    def _detect_base_pack_id(pack_ids: list[str]) -> str:
        vanilla_matches = [pack_id for pack_id in pack_ids if "vanilla" in pack_id.lower()]
        if len(vanilla_matches) == 1:
            return vanilla_matches[0]
        if len(vanilla_matches) > 1:
            raise ValueError(
                f"Found multiple vanilla-like packs {vanilla_matches}; cannot infer a unique base pack."
            )
        raise ValueError(
            f"Could not infer a base pack from {pack_ids}. Include exactly one pack name containing 'vanilla'."
        )

    def _build_support_filenames(self) -> dict[str, list[str]]:
        base_filenames = set(self.filename_to_index_per_pack.get(self.base_pack_id, {}))
        support_filenames_by_pack: dict[str, list[str]] = {}

        for pack_id, filename_to_index in self.filename_to_index_per_pack.items():
            if pack_id == self.base_pack_id:
                continue

            shared_filenames = sorted((base_filenames & set(filename_to_index)) - self.validation_filenames)
            if shared_filenames:
                support_filenames_by_pack[pack_id] = shared_filenames

        return support_filenames_by_pack

    def _support_pool_for_episode(self, target_pack: str, target_filename: str) -> list[str]:
        return [
            filename
            for filename in self.support_filenames_by_pack.get(target_pack, [])
            if filename != target_filename
        ]

    def _build_episodes(self, split_pairs: dict[str, list[list[str | int]]]) -> list[dict[str, str]]:
        base_lookup = self.filename_to_index_per_pack.get(self.base_pack_id, {})
        episodes: list[dict[str, str]] = []

        for filename in sorted(split_pairs):
            if filename not in base_lookup:
                continue

            target_packs = sorted(
                pack_id
                for pack_id, _array_idx in split_pairs[filename]
                if pack_id != self.base_pack_id and pack_id in self.filename_to_index_per_pack
            )

            for target_pack in target_packs:
                if not self._support_pool_for_episode(target_pack, filename):
                    continue
                episodes.append({"filename": filename, "target_pack": target_pack})

        if not episodes:
            raise ValueError(
                "No transfer episodes could be constructed. Re-run preprocessing and ensure the base pack "
                "shares non-validation textures with at least one target pack."
            )

        return episodes

    def __len__(self):
        return len(self.episodes)

    @staticmethod
    def _seed_from_episode(filename: str, target_pack: str, idx: int) -> int:
        text = f"{filename}:{target_pack}:{idx}"
        return sum((char_idx + 1) * ord(char) for char_idx, char in enumerate(text))

    def _episode_rng(self, filename: str, target_pack: str, idx: int) -> random.Random:
        if self.split == "train":
            return self._rng
        return random.Random(self._seed_from_episode(filename, target_pack, idx))

    def _sample_support_filenames(
        self,
        support_pool: list[str],
        rng: random.Random,
    ) -> list[str]:
        if not support_pool:
            raise ValueError("support_pool must not be empty")

        if len(support_pool) >= self.num_support_exemplars:
            return rng.sample(support_pool, k=self.num_support_exemplars)

        sampled = list(support_pool)
        while len(sampled) < self.num_support_exemplars:
            sampled.append(rng.choice(support_pool))
        return sampled

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        episode = self.episodes[idx]
        filename = episode["filename"]
        target_pack = episode["target_pack"]
        rng = self._episode_rng(filename, target_pack, idx)

        base_lookup = self.filename_to_index_per_pack[self.base_pack_id]
        target_lookup = self.filename_to_index_per_pack[target_pack]
        support_pool = self._support_pool_for_episode(target_pack, filename)
        support_filenames = self._sample_support_filenames(support_pool, rng)

        content_idx = base_lookup[filename]
        target_idx = target_lookup[filename]
        content_ref = torch.as_tensor(self.data[self.base_pack_id][content_idx], dtype=torch.long)
        target = torch.as_tensor(self.data[target_pack][target_idx], dtype=torch.long)

        support_content_refs = torch.stack(
            [
                torch.as_tensor(self.data[self.base_pack_id][base_lookup[support_filename]], dtype=torch.long)
                for support_filename in support_filenames
            ],
            dim=0,
        )
        support_style_refs = torch.stack(
            [
                torch.as_tensor(self.data[target_pack][target_lookup[support_filename]], dtype=torch.long)
                for support_filename in support_filenames
            ],
            dim=0,
        )

        return {
            "filename": filename,
            "content_filename": filename,
            "support_content_filenames": list(support_filenames),
            "support_style_filenames": list(support_filenames),
            "target_filename": filename,
            "content_pack": self.base_pack_id,
            "style_pack": target_pack,
            "target_pack": target_pack,
            "content_ref": content_ref,
            "support_content_refs": support_content_refs,
            "support_style_refs": support_style_refs,
            "target": target,
        }
