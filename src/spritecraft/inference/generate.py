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
    MIN_SUPPORT_EXEMPLARS,
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
    content_ref: torch.Tensor,
    support_content_refs: torch.Tensor,
    support_style_refs: torch.Tensor,
    target_ref: torch.Tensor,
) -> float:
    device = next(model.parameters()).device
    content_ref = content_ref.unsqueeze(0).to(device)
    support_content_refs = support_content_refs.unsqueeze(0).to(device)
    support_style_refs = support_style_refs.unsqueeze(0).to(device)
    target_ref = target_ref.unsqueeze(0).to(device)

    noisy_target = torch.full_like(target_ref, fill_value=MASK_TOKEN)
    t = torch.full((1,), NUM_TIMESTEPS, device=device, dtype=torch.long)
    logits = model(noisy_target, content_ref, support_content_refs, support_style_refs, None, t)
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


def _resolve_support_filenames(
    filename: str,
    target_pack: str,
    support_rankings: dict[str, dict[str, dict[str, list[str]]]],
    deterministic_supports: dict[str, dict[str, dict[str, list[str]]]],
    support_pool: list[str],
    support_count: int,
    rng: random.Random,
) -> list[str]:
    deterministic = deterministic_supports.get("val", {}).get(target_pack, {}).get(filename)
    if deterministic:
        return deterministic[:support_count]

    for split in ("val", "train"):
        ranked = support_rankings.get(split, {}).get(target_pack, {}).get(filename)
        if ranked:
            return ranked[:support_count]

    if not support_pool:
        return []

    if support_count >= len(support_pool):
        return list(support_pool)
    return rng.sample(support_pool, k=support_count)


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

    base_pack_id = pair_index.get("base_pack_id")
    if not isinstance(base_pack_id, str):
        raise ValueError("Pair index is missing base_pack_id; re-run preprocessing.")
    if pack_id not in filenames_per_pack:
        raise ValueError(f"Unknown pack id {pack_id!r}. Available packs: {sorted(filenames_per_pack)}")
    if base_pack_id not in filenames_per_pack:
        raise ValueError(f"Base pack {base_pack_id!r} not found in pair index; re-run preprocessing.")

    base_filenames = list(filenames_per_pack[base_pack_id])
    pack_filenames = set(filenames_per_pack[pack_id])
    support_pool_base = sorted(set(base_filenames) & pack_filenames)
    if not support_pool_base:
        raise ValueError(f"No shared textures between base pack {base_pack_id!r} and {pack_id!r}.")

    support_count = MIN_SUPPORT_EXEMPLARS
    support_count_range = pair_index.get("support_count_range")
    if isinstance(support_count_range, dict) and isinstance(support_count_range.get("min"), int):
        support_count = max(1, int(support_count_range["min"]))

    rng = random.Random(seed)
    selected_filenames = _select_texture_ids(textures, random_count, base_filenames, rng)

    support_rankings = pair_index.get("support_rankings", {})
    deterministic_supports = pair_index.get("deterministic_supports", {})
    filename_to_index = _build_filename_lookup(filenames_per_pack)

    dataset = np.load(DATASET_PATH)
    try:
        model, checkpoint_path = load_model(checkpoint)
        output_dir = Path(output_dir) / pack_id
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for filename in selected_filenames:
            content_ref = _lookup_tokens(dataset, filename_to_index, base_pack_id, filename)
            if content_ref is None:
                print(f"Skipping {filename}: missing in base pack {base_pack_id}")
                continue

            support_pool = [name for name in support_pool_base if name != filename]
            support_filenames = _resolve_support_filenames(
                filename=filename,
                target_pack=pack_id,
                support_rankings=support_rankings,
                deterministic_supports=deterministic_supports,
                support_pool=support_pool,
                support_count=min(support_count, len(support_pool)),
                rng=rng,
            )
            if not support_filenames:
                print(f"Skipping {filename}: no support pairs available for pack {pack_id}")
                continue

            support_content_refs = []
            support_style_refs = []
            for support_filename in support_filenames:
                support_content = _lookup_tokens(dataset, filename_to_index, base_pack_id, support_filename)
                support_style = _lookup_tokens(dataset, filename_to_index, pack_id, support_filename)
                if support_content is None or support_style is None:
                    continue
                support_content_refs.append(support_content)
                support_style_refs.append(support_style)

            if not support_content_refs:
                print(f"Skipping {filename}: could not resolve support tensors for {pack_id}")
                continue

            support_content_tensor = torch.stack(support_content_refs, dim=0)
            support_style_tensor = torch.stack(support_style_refs, dim=0)
            prediction = sample_tokens(
                model,
                content_ref,
                support_content_tensor,
                support_style_tensor,
            )

            truth_ref = _lookup_tokens(dataset, filename_to_index, pack_id, filename)
            extra_metrics = None
            if truth_ref is not None:
                cross_entropy = _compute_cross_entropy(
                    model,
                    content_ref,
                    support_content_tensor,
                    support_style_tensor,
                    truth_ref,
                )
                extra_metrics = {"cross_entropy": cross_entropy}

            bundle_name = f"{Path(filename).stem}_in_{pack_id}_{len(support_content_refs)}shot"
            result = save_prediction_bundle(
                output_dir=output_dir,
                bundle_name=bundle_name,
                content_tokens=content_ref,
                content_size=IMAGE_SIZE,
                support_content_tokens=support_content_refs,
                support_content_sizes=[IMAGE_SIZE] * len(support_content_refs),
                support_style_tokens=support_style_refs,
                support_style_sizes=[IMAGE_SIZE] * len(support_style_refs),
                prediction_tokens=prediction,
                prediction_size=IMAGE_SIZE,
                truth_tokens=truth_ref,
                truth_size=IMAGE_SIZE,
                metadata={
                    "filename": filename,
                    "content_pack": base_pack_id,
                    "target_pack": pack_id,
                    "support_filenames": support_filenames,
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
