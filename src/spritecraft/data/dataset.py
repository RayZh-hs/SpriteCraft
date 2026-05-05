"""Data loading and episodic texture-transfer dataset utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spritecraft.config import (
    DATASET_PATH,
    MAX_SUPPORT_EXEMPLARS,
    MIN_SUPPORT_EXEMPLARS,
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
        min_support_exemplars: int = MIN_SUPPORT_EXEMPLARS,
        max_support_exemplars: int = MAX_SUPPORT_EXEMPLARS,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        if min_support_exemplars <= 0:
            raise ValueError("min_support_exemplars must be positive")
        if max_support_exemplars < min_support_exemplars:
            raise ValueError("max_support_exemplars must be >= min_support_exemplars")

        self.split = split
        self.min_support_exemplars = min_support_exemplars
        self.max_support_exemplars = max_support_exemplars
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

        self.pack_roles = pair_data.get("pack_roles", {})
        self.pack_styles = pair_data.get("pack_styles", {})
        self.validation_filenames = set(pair_data.get("validation_filenames", sorted(VALIDATION_FILENAMES)))
        self.support_rankings = pair_data.get("support_rankings", {}).get(split, {})
        self.deterministic_supports = pair_data.get("deterministic_supports", {}).get(split, {})
        self.validation_matrix = [
            entry for entry in pair_data.get("validation_matrix", []) if entry.get("split") == split
        ]
        self.filename_to_index_per_pack = {
            pack_id: {filename: idx for idx, filename in enumerate(filenames)}
            for pack_id, filenames in self.filenames_per_pack.items()
        }
        self.episodes = self._build_episodes(split_pairs)
        self.episode_lookup = {
            (episode["filename"], episode["target_pack"]): idx
            for idx, episode in enumerate(self.episodes)
        }
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

    def _build_episodes(self, split_pairs: dict[str, list[list[str | int]]]) -> list[dict[str, str]]:
        base_lookup = self.filename_to_index_per_pack.get(self.base_pack_id, {})
        episodes: list[dict[str, str]] = []

        for filename in sorted(split_pairs):
            if filename not in base_lookup:
                continue

            target_packs = sorted(
                pack_id
                for pack_id, _ in split_pairs[filename]
                if isinstance(pack_id, str)
                and pack_id != self.base_pack_id
                and pack_id in self.filename_to_index_per_pack
            )

            for target_pack in target_packs:
                ranked_supports = self.support_rankings.get(target_pack, {}).get(filename, [])
                if not ranked_supports:
                    continue
                style = str(self.pack_styles.get(target_pack, "unspecified"))
                episodes.append(
                    {
                        "filename": filename,
                        "target_pack": target_pack,
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
        filename: str,
        target_pack: str,
        idx: int,
        rng: random.Random,
    ) -> list[str]:
        if self.split != "train":
            deterministic_supports = self.deterministic_supports.get(target_pack, {}).get(filename, [])
            if deterministic_supports:
                return list(deterministic_supports)

        ranked_supports = list(self.support_rankings.get(target_pack, {}).get(filename, []))
        if not ranked_supports:
            raise ValueError(f"No ranked supports available for {filename!r} in pack {target_pack!r}")

        actual_max = min(self.max_support_exemplars, len(ranked_supports))
        actual_min = min(self.min_support_exemplars, actual_max)
        if actual_max <= 0:
            raise ValueError(f"Support ranking for {filename!r} in {target_pack!r} is unexpectedly empty")
        support_count = rng.randint(actual_min, actual_max)

        retrieval_window = min(len(ranked_supports), max(self.max_support_exemplars * 2, support_count))
        candidate_pool = ranked_supports[:retrieval_window]
        if len(candidate_pool) <= support_count:
            return candidate_pool[:support_count]
        return rng.sample(candidate_pool, k=support_count)

    def _lookup_texture(self, pack_id: str, filename: str) -> torch.Tensor:
        array_idx = self.filename_to_index_per_pack[pack_id][filename]
        return torch.as_tensor(self.data[pack_id][array_idx], dtype=torch.long)

    def _padded_support_tensors(
        self,
        target_pack: str,
        support_filenames: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        support_content_refs = torch.zeros(
            (self.max_support_exemplars,) + self.data[self.base_pack_id].shape[1:],
            dtype=torch.long,
        )
        support_style_refs = torch.zeros(
            (self.max_support_exemplars,) + self.data[target_pack].shape[1:],
            dtype=torch.long,
        )
        support_mask = torch.zeros(self.max_support_exemplars, dtype=torch.bool)

        for support_idx, support_filename in enumerate(support_filenames):
            support_content_refs[support_idx] = self._lookup_texture(self.base_pack_id, support_filename)
            support_style_refs[support_idx] = self._lookup_texture(target_pack, support_filename)
            support_mask[support_idx] = True

        return support_content_refs, support_style_refs, support_mask

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
        rng = self._episode_rng(filename, target_pack, idx)
        support_filenames = self._sample_support_filenames(filename, target_pack, idx, rng)

        content_ref = self._lookup_texture(self.base_pack_id, filename)
        target = self._lookup_texture(target_pack, filename)
        support_content_refs, support_style_refs, support_mask = self._padded_support_tensors(
            target_pack,
            support_filenames,
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
            "style": episode.get("style", "unspecified"),
            "content_ref": content_ref,
            "support_content_refs": support_content_refs,
            "support_style_refs": support_style_refs,
            "support_mask": support_mask,
            "target": target,
        }
