"""Validation-set evaluation helpers for per-pack RGB models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

from PIL import Image
import torch

from spritecraft.config import (
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    OUTPUT_DIR,
    VALIDATION_MATRIX_EXAMPLES_PER_PACK,
    pack_checkpoint_dir,
)
from spritecraft.data.dataset import PackStyleDataset, get_available_pack_ids
from spritecraft.inference.sampler import load_model, sample_rgb, save_prediction_bundle
from spritecraft.models.unet import StyleAwareUNet


class SummaryOverall(TypedDict):
    examples: int
    mean_mae: float | None
    mean_pixel_accuracy: float | None
    exact_match_count: int


class Summary(TypedDict):
    split: str
    checkpoint_path: str | None
    overall: SummaryOverall
    packs: dict[str, dict[str, object]]


class PackResult(TypedDict):
    filename: str
    support_filenames: list[str]
    bundle_dir: str
    comparison_path: str
    metrics: dict[str, float | int | bool] | None


def _evaluate_sample(
    model: StyleAwareUNet,
    content_rgb: torch.Tensor,
    style_refs: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a single sample."""
    return sample_rgb(
        model,
        content_rgb.unsqueeze(0),
        style_refs,
        style_ref_mask=style_ref_mask,
        num_candidates=4,
    )


def _tile_images(images: list[Image.Image], columns: int = 2, gap: int = 6) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), color=(255, 255, 255))

    columns = max(1, columns)
    rows = (len(images) + columns - 1) // columns
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + gap * max(columns - 1, 0),
            rows * cell_height + gap * max(rows - 1, 0),
        ),
        color=(255, 255, 255),
    )

    for image_index, image in enumerate(images):
        row = image_index // columns
        column = image_index % columns
        x = column * (cell_width + gap)
        y = row * (cell_height + gap)
        canvas.paste(image, (x, y))

    return canvas


def write_validation_matrix(
    model: StyleAwareUNet,
    dataset: PackStyleDataset,
    output_dir: str | Path,
    checkpoint_path: Path | None = None,
) -> Summary:
    """Render a fixed validation matrix and per-pack summaries for a dataset split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use all validation samples (or a subset)
    num_samples = min(len(dataset), VALIDATION_MATRIX_EXAMPLES_PER_PACK * 4)
    
    results: list[PackResult] = []
    overall_metrics: list[dict[str, float | int | bool]] = []

    for idx in range(num_samples):
        sample = dataset[idx]
        filename = cast(str, sample["filename"])
        content_rgb = cast(torch.Tensor, sample["content_rgb"])
        target_rgb = cast(torch.Tensor, sample["target_rgb"])
        style_refs = cast(torch.Tensor, sample["style_refs"])
        style_ref_mask = cast(torch.Tensor, sample["style_ref_mask"])
        
        prediction = _evaluate_sample(model, content_rgb, style_refs, style_ref_mask=style_ref_mask)
        valid_support_count = int(style_ref_mask.sum().item())
        valid_supports = [cast(torch.Tensor, style_refs[i]) for i in range(valid_support_count)]

        bundle_name = (
            f"{dataset.split}_{idx:03d}_{Path(filename).stem}"
            f"_in_{dataset.pack_id}"
        )
        bundle_output_dir = output_dir / dataset.pack_id
        
        metadata: dict[str, str | int | float | bool | list[str]] = {
            "split": dataset.split,
            "index": idx,
            "filename": filename,
            "content_pack": dataset.base_pack_id,
            "target_pack": dataset.pack_id,
            "style": dataset.style,
        }
        if checkpoint_path is not None:
            metadata["checkpoint_path"] = str(checkpoint_path.resolve())

        result = save_prediction_bundle(
            output_dir=bundle_output_dir,
            bundle_name=bundle_name,
            content_rgb=content_rgb,
            content_size=IMAGE_SIZE,
            support_rgb=valid_supports,
            support_sizes=[IMAGE_SIZE] * len(valid_supports),
            prediction_rgb=prediction,
            prediction_size=IMAGE_SIZE,
            truth_rgb=target_rgb,
            truth_size=IMAGE_SIZE,
            metadata=metadata,
        )

        metrics = result["metrics"]
        if metrics is not None:
            overall_metrics.append(metrics)
        
        pack_entry: PackResult = {
            "filename": filename,
            "support_filenames": [],
            "bundle_dir": str(cast(Path, result["bundle_dir"])),
            "comparison_path": str(cast(Path, result["comparison_path"])),
            "metrics": metrics,
        }
        results.append(pack_entry)

    summary: Summary = {
        "split": dataset.split,
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path is not None else None,
        "overall": {
            "examples": len(overall_metrics),
            "mean_mae": (
                sum(float(metrics["mae"]) for metrics in overall_metrics) / len(overall_metrics)
                if overall_metrics
                else None
            ),
            "mean_pixel_accuracy": (
                sum(float(metrics["pixel_accuracy"]) for metrics in overall_metrics) / len(overall_metrics)
                if overall_metrics
                else None
            ),
            "exact_match_count": (
                sum(1 for metrics in overall_metrics if bool(metrics["exact_match"])) if overall_metrics else 0
            ),
        },
        "packs": {},
    }

    # Save pack summary
    pack_output_dir = output_dir / dataset.pack_id
    pack_output_dir.mkdir(parents=True, exist_ok=True)
    
    pack_images = []
    for pack_result in results:
        with Image.open(pack_result["comparison_path"]) as image:
            pack_images.append(image.copy())
    if pack_images:
        _tile_images(pack_images, columns=2).save(pack_output_dir / "summary.png")

    pack_metrics = [result["metrics"] for result in results if result["metrics"] is not None]
    pack_summary = {
        "pack": dataset.pack_id,
        "examples": len(results),
        "mean_mae": (
            sum(float(metrics["mae"]) for metrics in pack_metrics) / len(pack_metrics)
            if pack_metrics
            else None
        ),
        "mean_pixel_accuracy": (
            sum(float(metrics["pixel_accuracy"]) for metrics in pack_metrics) / len(pack_metrics)
            if pack_metrics
            else None
        ),
        "exact_match_count": sum(1 for metrics in pack_metrics if bool(metrics["exact_match"])),
        "entries": results,
    }
    summary["packs"][dataset.pack_id] = pack_summary
    
    with open(pack_output_dir / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(pack_summary, file_obj, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    
    return summary


def run(
    checkpoint_dir: str | Path = CHECKPOINTS_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    split: str = "val",
    mode: str = "matrix",
    pack_id: str | None = None,
):
    """Evaluate either the fixed validation matrix or one dataset example."""
    if mode not in {"matrix", "single"}:
        raise ValueError(f"Unsupported evaluation mode {mode!r}")

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)

    if pack_id is not None:
        # Evaluate specific pack
        pack_ckpt = pack_checkpoint_dir(checkpoint_dir, pack_id)
        if not pack_ckpt.exists():
            print(f"No checkpoint found for pack {pack_id}")
            return output_dir
        
        model, ckpt_path = load_model(pack_ckpt)
        dataset = PackStyleDataset(pack_id=pack_id, split=split)
        
        if len(dataset) == 0:
            print(f"Dataset split {split!r} is empty for pack {pack_id}")
            return output_dir
        
        write_validation_matrix(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            checkpoint_path=ckpt_path,
        )
        print(f"Saved validation matrix for pack={pack_id} split={split} to {output_dir.resolve()}")
        return output_dir
    
    # Evaluate all available packs
    available_packs = get_available_pack_ids()
    if not available_packs:
        raise ValueError("No preprocessed pack datasets found. Run preprocessing first.")
    
    for pack_id in available_packs:
        pack_ckpt = pack_checkpoint_dir(checkpoint_dir, pack_id)
        if not pack_ckpt.exists():
            print(f"Skipping {pack_id}: no checkpoint found")
            continue
        
        try:
            model, ckpt_path = load_model(pack_ckpt)
            dataset = PackStyleDataset(pack_id=pack_id, split=split)
            
            if len(dataset) == 0:
                print(f"Skipping {pack_id}: dataset split {split!r} is empty")
                continue
            
            write_validation_matrix(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                checkpoint_path=ckpt_path,
            )
            print(f"Saved validation matrix for pack={pack_id} split={split}")
        except Exception as exc:
            print(f"Error evaluating {pack_id}: {exc}")
            continue
    
    return output_dir
