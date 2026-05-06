"""Unified texture generation for resource-pack evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from spritecraft.config import (
    CHECKPOINTS_DIR,
    DATASET_PATH,
    IMAGE_SIZE,
    MASK_TOKEN,
    NUM_TIMESTEPS,
    OUTPUT_DIR,
    PAIR_INDEX_PATH,
)
from spritecraft.inference.sampler import load_model, sample_tokens, save_prediction_bundle


def _normalize_texture_id(texture_id: str) -> str:
    name = Path(texture_id).name
    if not name.lower().endswith(".png"):
        name = f"{name}.png"
    return name


def _load_pair_index(pair_index_path: Path) -> dict[str, Any]:
    if not pair_index_path.exists():
        raise FileNotFoundError(f"Pair index not found at {pair_index_path}. Run preprocessing first.")
    with pair_index_path.open(encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError("Pair index is malformed; expected a JSON object.")
    return data


def _build_filename_lookup(filenames_per_pack: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    return {
        pack_id: {filename: idx for idx, filename in enumerate(filenames)}
        for pack_id, filenames in filenames_per_pack.items()
    }


@torch.no_grad()
def _compute_cross_entropy(
    model: torch.nn.Module,
    source: torch.Tensor,
    pack_id: torch.Tensor,
    target_ref: torch.Tensor,
) -> float:
    device = next(model.parameters()).device
    source = source.unsqueeze(0).to(device)
    pack_id = pack_id.unsqueeze(0).to(device)
    target_ref = target_ref.unsqueeze(0).to(device)

    noisy_target = torch.full_like(target_ref, fill_value=MASK_TOKEN)
    t = torch.full((1,), NUM_TIMESTEPS, device=device, dtype=torch.long)
    logits = model(noisy_target, source, pack_id, t)
    loss = F.cross_entropy(logits, target_ref)
    return float(loss.detach().cpu().item())


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


def _lookup_tokens(
    dataset: np.lib.npyio.NpzFile,
    filename_to_index: dict[str, dict[str, int]],
    pack_id: str,
    filename: str,
) -> torch.Tensor | None:
    array_idx = filename_to_index.get(pack_id, {}).get(filename)
    if array_idx is None:
        return None
    return torch.as_tensor(dataset[pack_id][array_idx], dtype=torch.long)


def run(
    pack_id: str,
    checkpoint: str | Path = CHECKPOINTS_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    textures: list[str] | None = None,
    random_count: int | None = None,
    seed: int | None = None,
):
    """Generate textures for a target pack using preprocessed dataset assets."""
    pair_index = _load_pair_index(Path(PAIR_INDEX_PATH))
    filenames_per_pack = pair_index.get("filenames_per_pack", {})
    if not isinstance(filenames_per_pack, dict) or not filenames_per_pack:
        raise ValueError("Pair index is missing filenames_per_pack; re-run preprocessing.")

    pack_ids = pair_index.get("pack_ids", [])
    if not pack_ids:
        raise ValueError("Pair index is missing pack_ids; re-run preprocessing.")

    pack_name_to_idx = {name: idx for idx, name in enumerate(pack_ids)}
    base_pack_idx = pair_index.get("base_pack_idx")
    if not isinstance(base_pack_idx, int):
        raise ValueError("Pair index is missing base_pack_idx; re-run preprocessing.")

    base_pack_id = pack_ids[base_pack_idx]
    if pack_id not in filenames_per_pack:
        raise ValueError(f"Unknown pack id {pack_id!r}. Available packs: {sorted(filenames_per_pack)}")
    if base_pack_id not in filenames_per_pack:
        raise ValueError(f"Base pack {base_pack_id!r} not found in pair index; re-run preprocessing.")

    pack_idx = pack_name_to_idx[pack_id]
    base_filenames = list(filenames_per_pack[base_pack_id])

    rng = random.Random(seed)
    selected_filenames = _select_texture_ids(textures, random_count, base_filenames, rng)

    filename_to_index = _build_filename_lookup(filenames_per_pack)

    dataset = np.load(DATASET_PATH)
    try:
        model, checkpoint_path = load_model(checkpoint)
        output_dir = Path(output_dir) / pack_id
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for filename in selected_filenames:
            source = _lookup_tokens(dataset, filename_to_index, base_pack_id, filename)
            if source is None:
                print(f"Skipping {filename}: missing in base pack {base_pack_id}")
                continue

            prediction = sample_tokens(
                model,
                source,
                torch.tensor(pack_idx, dtype=torch.long),
            )

            truth_ref = _lookup_tokens(dataset, filename_to_index, pack_id, filename)
            extra_metrics = None
            if truth_ref is not None:
                cross_entropy = _compute_cross_entropy(
                    model,
                    source,
                    torch.tensor(pack_idx, dtype=torch.long),
                    truth_ref,
                )
                extra_metrics = {"cross_entropy": cross_entropy}

            bundle_name = f"{Path(filename).stem}_in_{pack_id}"
            result = save_prediction_bundle(
                output_dir=output_dir,
                bundle_name=bundle_name,
                source_tokens=source,
                source_size=IMAGE_SIZE,
                prediction_tokens=prediction,
                prediction_size=IMAGE_SIZE,
                pack_id=pack_idx,
                truth_tokens=truth_ref,
                truth_size=IMAGE_SIZE,
                metadata={
                    "filename": filename,
                    "source_pack": base_pack_id,
                    "target_pack": pack_id,
                    "pack_idx": pack_idx,
                    "checkpoint_path": str(Path(checkpoint_path).resolve()),
                    "target_available": truth_ref is not None,
                },
                extra_metrics=extra_metrics,
            )
            results.append((filename, result))

            print(f"Generated {filename} -> {result['produced_path']}")
            metrics = result["metrics"]
            if isinstance(metrics, dict) and "cross_entropy" in metrics:
                print(f"  cross_entropy={metrics['cross_entropy']:.4f}")

        if not results:
            raise RuntimeError("No textures were generated. Check your selection and pack data.")
        return [result["bundle_dir"] for _filename, result in results]
    finally:
        dataset.close()
