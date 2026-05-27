"""Unified texture generation for resource-pack evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spritecraft.config import (
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    MAX_SUPPORT_EXEMPLARS,
    OUTPUT_DIR,
    pack_dataset_path,
    pack_pair_index_path,
)
from spritecraft.data.support_index import compute_texture_descriptor, rank_support_candidates
from spritecraft.inference.sampler import compute_metrics, load_model, sample_rgb, save_prediction_bundle


def _normalize_texture_id(texture_id: str) -> str:
    name = Path(texture_id).name
    if not name.lower().endswith(".png"):
        name = f"{name}.png"
    return name


def _load_pair_index(pack_id: str) -> dict[str, Any]:
    pair_index_path = pack_pair_index_path(pack_id)
    if not pair_index_path.exists():
        raise FileNotFoundError(f"Pair index not found for pack {pack_id}: {pair_index_path}. Run preprocessing first.")
    with pair_index_path.open(encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError("Pair index is malformed; expected a JSON object.")
    return data


def _load_dataset(pack_id: str) -> np.lib.npyio.NpzFile:
    dataset_path = pack_dataset_path(pack_id)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found for pack {pack_id}: {dataset_path}. Run preprocessing first.")
    return np.load(dataset_path)


def _select_texture_ids(
    textures: list[str] | None,
    random_count: int | None,
    base_filenames: list[str],
    rng: random.Random,
) -> list[str]:
    base_set = set(base_filenames)
    if textures is not None:
        normalized = [_normalize_texture_id(texture_id) for texture_id in textures]
        missing = [texture_id for texture_id in normalized if texture_id not in base_set]
        if missing:
            raise ValueError(f"Unknown texture ids (not in base pack): {missing}")
        return normalized

    if random_count is None:
        raise ValueError("Either textures or random_count must be provided")
    if random_count <= 0:
        raise ValueError("random_count must be positive")
    if random_count > len(base_filenames):
        raise ValueError(f"random_count exceeds available base textures ({len(base_filenames)})")

    return rng.sample(base_filenames, k=random_count)


def _resolve_support_filenames(
    filename: str,
    train_filenames: list[str],
    content_descriptors: dict[str, np.ndarray],
    support_count: int = MAX_SUPPORT_EXEMPLARS,
) -> list[str]:
    available = [f for f in train_filenames if f != filename]
    if not available:
        return []

    ranked = rank_support_candidates(filename, available, content_descriptors)
    if not ranked:
        ranked = sorted(available)
    count = min(support_count, len(ranked))
    return ranked[:count]


def _build_content_descriptors(
    train_filenames: list[str],
    val_filenames: list[str],
    dataset: np.lib.npyio.NpzFile,
) -> dict[str, np.ndarray]:
    descriptors: dict[str, np.ndarray] = {}

    for filename, image in zip(train_filenames, dataset["content_rgb_train"], strict=False):
        descriptors[filename] = compute_texture_descriptor((image * 255.0).clip(0, 255).astype(np.uint8))

    for filename, image in zip(val_filenames, dataset["content_rgb_val"], strict=False):
        descriptors[filename] = compute_texture_descriptor((image * 255.0).clip(0, 255).astype(np.uint8))

    return descriptors


def run(
    pack_id: str,
    checkpoint: str | Path = CHECKPOINTS_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    textures: list[str] | None = None,
    random_count: int | None = None,
    seed: int | None = None,
):
    """Generate textures for a target pack using preprocessed dataset assets."""
    pair_index = _load_pair_index(pack_id)
    base_pack_id = pair_index.get("base_pack_id")
    
    if not isinstance(base_pack_id, str):
        raise ValueError("Pair index is missing base_pack_id; re-run preprocessing.")

    # Load all available filenames
    train_filenames = pair_index.get("train_filenames", [])
    val_filenames = pair_index.get("val_filenames", [])
    all_target_filenames = pair_index.get("all_target_filenames", [])
    base_filenames = train_filenames + val_filenames
    
    if not base_filenames:
        raise ValueError(f"No shared textures found for pack {pack_id}. Run preprocessing first.")

    support_count = MAX_SUPPORT_EXEMPLARS
    rng = random.Random(seed)
    selected_filenames = _select_texture_ids(textures, random_count, base_filenames, rng)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Load dataset
    dataset = _load_dataset(pack_id)
    
    try:
        model, checkpoint_path = load_model(checkpoint, pack_id)
        output_dir = Path(output_dir) / pack_id
        output_dir.mkdir(parents=True, exist_ok=True)
        content_descriptors = _build_content_descriptors(train_filenames, val_filenames, dataset)

        results = []
        for filename in selected_filenames:
            # Find index of this filename in train/val
            try:
                if filename in train_filenames:
                    idx = train_filenames.index(filename)
                    content_rgb = torch.from_numpy(dataset["content_rgb_train"][idx]).float()
                    target_rgb = torch.from_numpy(dataset["target_rgb_train"][idx]).float()
                else:
                    idx = val_filenames.index(filename)
                    content_rgb = torch.from_numpy(dataset["content_rgb_val"][idx]).float()
                    target_rgb = torch.from_numpy(dataset["target_rgb_val"][idx]).float()
            except (ValueError, KeyError, IndexError):
                print(f"Skipping {filename}: not found in dataset")
                continue

            # Permute to [C, H, W]
            content_rgb = content_rgb.permute(2, 0, 1)  # [3, 32, 32]
            target_rgb = target_rgb.permute(2, 0, 1)  # [3, 32, 32]

            # Sample style references
            support_filenames = _resolve_support_filenames(
                filename=filename,
                train_filenames=train_filenames,
                content_descriptors=content_descriptors,
                support_count=support_count,
            )
            
            if not support_filenames:
                print(f"Skipping {filename}: no support pairs available")
                continue

            # Load style reference RGBs
            support_rgb_list = []
            support_content_list = []
            support_indices = [all_target_filenames.index(f) for f in support_filenames]
            all_target_rgb = dataset["all_target_rgb"]
            for support_filename, sidx in zip(support_filenames, support_indices, strict=True):
                srgb = torch.from_numpy(all_target_rgb[sidx]).float().permute(2, 0, 1)
                support_rgb_list.append(srgb)
                source_idx = train_filenames.index(support_filename)
                source_rgb = torch.from_numpy(dataset["content_rgb_train"][source_idx]).float().permute(2, 0, 1)
                support_content_list.append(source_rgb)
            
            support_rgb_tensor = torch.stack(support_rgb_list, dim=0)  # [N, 3, 32, 32]
            support_content_tensor = torch.stack(support_content_list, dim=0)  # [N, 3, 32, 32]
            style_ref_mask = torch.ones(len(support_rgb_list), dtype=torch.bool)
            
            # Generate
            prediction = sample_rgb(
                model,
                content_rgb,
                support_rgb_tensor,
                style_ref_mask=style_ref_mask,
                support_content_refs=support_content_tensor,
                num_candidates=4,
            )

            # Compute metrics if target is available
            extra_metrics = None
            try:
                metrics = compute_metrics(prediction, target_rgb)
                extra_metrics = metrics
            except Exception:
                pass

            bundle_name = f"{Path(filename).stem}_in_{pack_id}_{len(support_rgb_list)}shot"
            result = save_prediction_bundle(
                output_dir=output_dir,
                bundle_name=bundle_name,
                content_rgb=content_rgb,
                content_size=IMAGE_SIZE,
                support_rgb=support_rgb_list,
                support_sizes=[IMAGE_SIZE] * len(support_rgb_list),
                prediction_rgb=prediction,
                prediction_size=IMAGE_SIZE,
                truth_rgb=target_rgb,
                truth_size=IMAGE_SIZE,
                metadata={
                    "filename": filename,
                    "content_pack": base_pack_id,
                    "target_pack": pack_id,
                    "support_filenames": support_filenames,
                    "checkpoint_path": str(Path(checkpoint_path).resolve()),
                    "target_available": True,
                },
                extra_metrics=extra_metrics,
            )
            results.append((filename, result))

            print(f"Generated {filename} -> {result['produced_path']}")
            metrics = result["metrics"]
            if isinstance(metrics, dict) and "mae" in metrics:
                print(f"  mae={metrics['mae']:.4f} pixel_accuracy={metrics['pixel_accuracy']:.4f}")

        if not results:
            raise RuntimeError("No textures were generated. Check your selection and pack data.")
        return [result["bundle_dir"] for _filename, result in results]
    finally:
        dataset.close()
