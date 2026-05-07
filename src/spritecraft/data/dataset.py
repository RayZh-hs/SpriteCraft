"""Per-pack RGB dataset with style reference sampling."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spritecraft.config import (
    IMAGE_SIZE,
    MAX_SUPPORT_EXEMPLARS,
    MIN_SUPPORT_EXEMPLARS,
    pack_dataset_path,
    pack_pair_index_path,
    VALIDATION_FILENAMES,
)


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
            if split == "train":
                self.content_rgb = torch.from_numpy(data["content_rgb_train"]).float()
                self.target_rgb = torch.from_numpy(data["target_rgb_train"]).float()
                self.content_alpha = torch.from_numpy(data["content_alpha_train"]).float()
                self.target_alpha = torch.from_numpy(data["target_alpha_train"]).float()
            else:
                self.content_rgb = torch.from_numpy(data["content_rgb_val"]).float()
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
        
        if len(self.filenames) == 0:
            raise ValueError(f"No {split} samples found for pack {pack_id}")
        
        # Build mapping from filename to index in all_target arrays
        self.filename_to_all_idx = {
            filename: idx for idx, filename in enumerate(self.all_target_filenames)
        }

    def __len__(self) -> int:
        return len(self.filenames)

    def _sample_style_refs(self, exclude_filename: str, idx: int) -> torch.Tensor:
        """Sample style reference textures from the target pack."""
        # During validation, only use training textures as style references
        # to avoid data leakage
        if self.split == "val":
            available = [
                filename for filename in self.train_filenames
                if filename != exclude_filename
            ]
        else:
            # During training, can use any texture except current
            available = [
                filename for filename in self.all_target_filenames
                if filename != exclude_filename
            ]
        
        if not available:
            # Fallback: use current texture if no others available
            available = [exclude_filename]
        
        # Determine number of style references
        if self.split == "val":
            # Deterministic for validation
            num_refs = min(self.max_style_refs, len(available))
            selected = available[:num_refs]
        else:
            num_refs = self._rng.randint(self.min_style_refs, min(self.max_style_refs, len(available)))
            selected = self._rng.sample(available, k=num_refs)
        
        # Gather RGB arrays
        ref_indices = [self.filename_to_all_idx[f] for f in selected]
        style_refs = self.all_target_rgb[ref_indices]  # [N, 32, 32, 3]
        
        # Pad to max_style_refs if needed
        if style_refs.shape[0] < self.max_style_refs:
            padding = torch.zeros(
                self.max_style_refs - style_refs.shape[0],
                IMAGE_SIZE, IMAGE_SIZE, 3,
                dtype=torch.float32,
            )
            style_refs = torch.cat([style_refs, padding], dim=0)
        
        return style_refs

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
        style_refs = self._sample_style_refs(filename, idx)  # [max_refs, 32, 32, 3]
        
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
