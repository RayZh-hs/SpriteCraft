"""Per-pack RGB dataset with style reference sampling."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from spritecraft.config import (
    IMAGE_SIZE,
    MAX_SUPPORT_EXEMPLARS,
    MIN_SUPPORT_EXEMPLARS,
    pack_dataset_path,
    pack_pair_index_path,
)
from spritecraft.data.support_index import compute_texture_descriptor, rank_support_candidates


class PackStyleDataset(Dataset):
    """PyTorch dataset for per-pack RGB style transfer.
    
    Loads preprocessed per-pack RGB data and samples style references
    from the same target pack.
    """

    def __init__(
        self,
        pack_id: str,
        split: str = "train",
        seed: int = 42,
        min_style_refs: int = MIN_SUPPORT_EXEMPLARS,
        max_style_refs: int = MAX_SUPPORT_EXEMPLARS,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        self.pack_id = pack_id
        self.split = split
        self.min_style_refs = min_style_refs
        self.max_style_refs = max_style_refs
        self._rng = random.Random(seed)

        # Load dataset
        dataset_path = pack_dataset_path(pack_id)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found for pack {pack_id}: {dataset_path}")

        with np.load(dataset_path) as data:
            self._content_rgb_train = torch.from_numpy(data["content_rgb_train"]).float()
            self._content_rgb_val = torch.from_numpy(data["content_rgb_val"]).float()
            if split == "train":
                self.content_rgb = self._content_rgb_train
                self.target_rgb = torch.from_numpy(data["target_rgb_train"]).float()
                self.content_alpha = torch.from_numpy(data["content_alpha_train"]).float()
                self.target_alpha = torch.from_numpy(data["target_alpha_train"]).float()
            else:
                self.content_rgb = self._content_rgb_val
                self.target_rgb = torch.from_numpy(data["target_rgb_val"]).float()
                self.content_alpha = torch.from_numpy(data["content_alpha_val"]).float()
                self.target_alpha = torch.from_numpy(data["target_alpha_val"]).float()
            
            # All target textures for style reference sampling
            self.all_target_rgb = torch.from_numpy(data["all_target_rgb"]).float()
            self.all_target_alpha = torch.from_numpy(data["all_target_alpha"]).float()

        # Load pair index
        pair_index_path = pack_pair_index_path(pack_id)
        with open(pair_index_path, encoding="utf-8") as f:
            pair_index = json.load(f)
        
        self.base_pack_id = pair_index["base_pack_id"]
        self.style = pair_index.get("style", "unspecified")
        
        if split == "train":
            self.filenames = pair_index["train_filenames"]
        else:
            self.filenames = pair_index["val_filenames"]
        
        self.all_target_filenames = pair_index["all_target_filenames"]
        self.train_filenames = pair_index["train_filenames"]
        self.val_filenames = pair_index["val_filenames"]
        self.shared_filenames = self.train_filenames + self.val_filenames
        
        if len(self.filenames) == 0:
            raise ValueError(f"No {split} samples found for pack {pack_id}")
        
        # Build mapping from filename to index in all_target arrays
        self.filename_to_all_idx = {
            filename: idx for idx, filename in enumerate(self.all_target_filenames)
        }
        self.content_descriptors = self._build_content_descriptors()
        self.support_rankings = self._build_support_rankings()

    def __len__(self) -> int:
        return len(self.filenames)

    def _build_content_descriptors(self) -> dict[str, np.ndarray]:
        descriptors: dict[str, np.ndarray] = {}

        for filename, tensor in zip(self.train_filenames, self._content_rgb_train, strict=False):
            image = (tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
            descriptors[filename] = compute_texture_descriptor(image)

        for filename, tensor in zip(self.val_filenames, self._content_rgb_val, strict=False):
            image = (tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
            descriptors[filename] = compute_texture_descriptor(image)

        return descriptors

    def _build_support_rankings(self) -> dict[str, list[str]]:
        rankings: dict[str, list[str]] = {}
        for filename in self.shared_filenames:
            candidates = [
                candidate for candidate in self.train_filenames
                if candidate != filename
            ]
            ranked = rank_support_candidates(filename, candidates, self.content_descriptors)
            rankings[filename] = ranked if ranked else sorted(candidates)
        return rankings

    def _sample_style_refs(self, exclude_filename: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample style reference textures from the target pack."""
        ranked_available = self.support_rankings.get(exclude_filename, [])

        if not ranked_available and self.split == "train":
            # Training fallback for tiny packs with only one train texture.
            ranked_available = [exclude_filename]

        max_refs = min(self.max_style_refs, len(ranked_available))
        min_refs = min(self.min_style_refs, max_refs)

        if self.split == "val":
            selected = ranked_available[:max_refs]
        else:
            num_refs = self._rng.randint(min_refs, max_refs)
            candidate_pool = ranked_available[: max(num_refs, min(len(ranked_available), num_refs * 2))]
            selected = self._rng.sample(candidate_pool, k=num_refs)

        # Gather RGB arrays
        ref_indices = [self.filename_to_all_idx[f] for f in selected]
        style_refs = self.all_target_rgb[ref_indices]  # [N, 32, 32, 3]
        style_ref_mask = torch.ones(style_refs.shape[0], dtype=torch.bool)
        
        # Pad to max_style_refs if needed
        if style_refs.shape[0] < self.max_style_refs:
            padding = torch.zeros(
                self.max_style_refs - style_refs.shape[0],
                IMAGE_SIZE, IMAGE_SIZE, 3,
                dtype=torch.float32,
            )
            style_refs = torch.cat([style_refs, padding], dim=0)
            style_ref_mask = torch.cat(
                [
                    style_ref_mask,
                    torch.zeros(self.max_style_refs - style_ref_mask.shape[0], dtype=torch.bool),
                ],
                dim=0,
            )

        return style_refs, style_ref_mask

    def __getitem__(self, idx: int) -> dict[str, object]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        filename = self.filenames[idx]
        
        # Get content (vanilla) and target (styled) RGB
        content_rgb = self.content_rgb[idx]  # [32, 32, 3]
        target_rgb = self.target_rgb[idx]  # [32, 32, 3]
        content_alpha = self.content_alpha[idx]  # [32, 32]
        target_alpha = self.target_alpha[idx]  # [32, 32]
        
        # Sample style references from target pack
        style_refs, style_ref_mask = self._sample_style_refs(filename)  # [max_refs, 32, 32, 3]
        
        # Permute to [C, H, W] format
        content_rgb = content_rgb.permute(2, 0, 1)  # [3, 32, 32]
        target_rgb = target_rgb.permute(2, 0, 1)  # [3, 32, 32]
        style_refs = style_refs.permute(0, 3, 1, 2)  # [max_refs, 3, 32, 32]
        
        return {
            "filename": filename,
            "content_rgb": content_rgb,
            "target_rgb": target_rgb,
            "content_alpha": content_alpha,
            "target_alpha": target_alpha,
            "style_refs": style_refs,
            "style_ref_mask": style_ref_mask,
            "pack_id": self.pack_id,
            "base_pack_id": self.base_pack_id,
            "style": self.style,
        }


def get_available_pack_ids() -> list[str]:
    """Get list of pack IDs that have preprocessed datasets."""
    if not pack_dataset_path("").parent.exists():
        return []
    
    pack_ids = []
    for pack_dir in pack_dataset_path("").parent.iterdir():
        if pack_dir.is_dir():
            dataset_file = pack_dir / "dataset.npz"
            pair_index_file = pack_dir / "pair_index.json"
            if dataset_file.exists() and pair_index_file.exists():
                pack_ids.append(pack_dir.name)
    
    return sorted(pack_ids)
