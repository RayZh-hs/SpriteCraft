"""Validation-set evaluation helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from spritecraft.config import CHECKPOINTS_DIR, IMAGE_SIZE, OUTPUT_DIR, VALIDATION_MATRIX_EXAMPLES_PER_PACK
from spritecraft.data.dataset import TextureDataset
from spritecraft.inference.sampler import load_model, sample_tokens, save_prediction_bundle
from spritecraft.models.unet import UNet


def _resolve_index(
    dataset: TextureDataset,
    index: int,
    filename: str | None,
    target_pack: str | None,
) -> int:
    if filename is not None and target_pack is not None:
        return dataset.get_episode_index(filename, target_pack)

    if filename is not None:
        for episode_idx, episode in enumerate(dataset.episodes):
            if episode["filename"] == filename:
                return episode_idx
        raise ValueError(f"{filename!r} is not present in the {dataset.split} split")

    if index < 0 or index >= len(dataset):
        raise IndexError(f"index {index} is out of range for split {dataset.split!r} (size={len(dataset)})")
    return index


def _active_support_count(sample: dict[str, object]) -> int:
    return int(sample["support_mask"].sum().item())


def _trimmed_support_data(sample: dict[str, object]) -> tuple[list, list, list, list]:
    support_count = _active_support_count(sample)
    return (
        list(sample["support_content_refs"][:support_count]),
        [IMAGE_SIZE] * support_count,
        list(sample["support_style_refs"][:support_count]),
        [IMAGE_SIZE] * support_count,
    )


def _evaluate_sample(model: UNet, sample: dict[str, object]):
    return sample_tokens(
        model,
        sample["content_ref"].unsqueeze(0),
        sample["support_content_refs"].unsqueeze(0),
        sample["support_style_refs"].unsqueeze(0),
        support_mask=sample["support_mask"].unsqueeze(0),
    ).squeeze(0)


def _tile_images(images: list[Image.Image], columns: int = 2) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), color=(255, 255, 255))

    columns = max(1, columns)
    rows = (len(images) + columns - 1) // columns
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), color=(255, 255, 255))

    for image_index, image in enumerate(images):
        row = image_index // columns
        column = image_index % columns
        x = column * cell_width
        y = row * cell_height
        canvas.paste(image, (x, y))

    return canvas


def _fallback_matrix_entries(dataset: TextureDataset) -> list[dict[str, str]]:
    entries_by_pack: dict[str, list[dict[str, str]]] = defaultdict(list)
    for episode in dataset.episodes:
        entries_by_pack[episode["target_pack"]].append(
            {
                "split": dataset.split,
                "filename": episode["filename"],
                "target_pack": episode["target_pack"],
            }
        )

    entries: list[dict[str, str]] = []
    for target_pack in sorted(entries_by_pack):
        entries.extend(entries_by_pack[target_pack][:VALIDATION_MATRIX_EXAMPLES_PER_PACK])
    return entries


def write_validation_matrix(
    model: UNet,
    dataset: TextureDataset,
    output_dir: str | Path,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Render a fixed validation matrix and per-pack summaries for a dataset split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_entries = list(dataset.validation_matrix) if dataset.validation_matrix else _fallback_matrix_entries(dataset)
    if not matrix_entries:
        raise ValueError(f"No validation matrix entries are available for split {dataset.split!r}")

    results_by_pack: dict[str, list[dict[str, object]]] = defaultdict(list)
    overall_metrics: list[dict[str, float | int | bool]] = []

    for entry in matrix_entries:
        dataset_index = dataset.get_episode_index(entry["filename"], entry["target_pack"])
        sample = dataset[dataset_index]
        prediction = _evaluate_sample(model, sample)
        support_count = _active_support_count(sample)
        support_content_tokens, support_content_sizes, support_style_tokens, support_style_sizes = _trimmed_support_data(sample)

        bundle_name = (
            f"{dataset.split}_{dataset_index:03d}_{Path(sample['filename']).stem}"
            f"_in_{sample['target_pack']}"
            f"_{support_count}shot"
        )
        bundle_output_dir = output_dir / sample["target_pack"]
        result = save_prediction_bundle(
            output_dir=bundle_output_dir,
            bundle_name=bundle_name,
            content_tokens=sample["content_ref"],
            content_size=IMAGE_SIZE,
            support_content_tokens=support_content_tokens,
            support_content_sizes=support_content_sizes,
            support_style_tokens=support_style_tokens,
            support_style_sizes=support_style_sizes,
            prediction_tokens=prediction,
            prediction_size=IMAGE_SIZE,
            truth_tokens=sample["target"],
            truth_size=IMAGE_SIZE,
            metadata={
                "split": dataset.split,
                "index": dataset_index,
                "filename": sample["filename"],
                "content_filename": sample["content_filename"],
                "support_content_filenames": sample["support_content_filenames"],
                "support_style_filenames": sample["support_style_filenames"],
                "target_filename": sample["target_filename"],
                "content_pack": sample["content_pack"],
                "style_pack": sample["style_pack"],
                "target_pack": sample["target_pack"],
                "style": sample.get("style", "unspecified"),
                "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path is not None else None,
            },
        )

        metrics = result["metrics"]
        if metrics is not None:
            overall_metrics.append(metrics)
        results_by_pack[sample["target_pack"]].append(
            {
                "filename": sample["filename"],
                "support_filenames": sample["support_style_filenames"],
                "bundle_dir": str(result["bundle_dir"]),
                "comparison_path": str(result["comparison_path"]),
                "metrics": metrics,
            }
        )

    summary = {
        "split": dataset.split,
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path is not None else None,
        "overall": {
            "examples": len(overall_metrics),
            "mean_pixel_accuracy": (
                sum(float(metrics["pixel_accuracy"]) for metrics in overall_metrics) / len(overall_metrics)
                if overall_metrics
                else None
            ),
            "mean_rgb_mae": (
                sum(float(metrics["rgb_mae"]) for metrics in overall_metrics) / len(overall_metrics)
                if overall_metrics
                else None
            ),
            "exact_match_count": (
                sum(1 for metrics in overall_metrics if bool(metrics["exact_match"])) if overall_metrics else 0
            ),
        },
        "packs": {},
    }

    for target_pack, pack_results in sorted(results_by_pack.items()):
        pack_output_dir = output_dir / target_pack
        pack_images = []
        for result in pack_results:
            with Image.open(result["comparison_path"]) as image:
                pack_images.append(image.copy())
        _tile_images(pack_images, columns=2).save(pack_output_dir / "summary.png")

        pack_metrics = [result["metrics"] for result in pack_results if result["metrics"] is not None]
        pack_summary = {
            "pack": target_pack,
            "examples": len(pack_results),
            "mean_pixel_accuracy": (
                sum(float(metrics["pixel_accuracy"]) for metrics in pack_metrics) / len(pack_metrics)
                if pack_metrics
                else None
            ),
            "mean_rgb_mae": (
                sum(float(metrics["rgb_mae"]) for metrics in pack_metrics) / len(pack_metrics)
                if pack_metrics
                else None
            ),
            "exact_match_count": sum(1 for metrics in pack_metrics if bool(metrics["exact_match"])),
            "entries": pack_results,
        }
        summary["packs"][target_pack] = pack_summary
        with open(pack_output_dir / "summary.json", "w", encoding="utf-8") as file_obj:
            json.dump(pack_summary, file_obj, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    return summary


def _run_single(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    split: str,
    index: int,
    filename: str | None,
    target_pack: str | None,
):
    dataset = TextureDataset(split=split)
    if len(dataset) == 0:
        raise ValueError(f"Dataset split {split!r} is empty. Run preprocessing first.")

    dataset_index = _resolve_index(dataset, index=index, filename=filename, target_pack=target_pack)
    sample = dataset[dataset_index]
    model, checkpoint_path = load_model(checkpoint_dir)
    prediction = _evaluate_sample(model, sample)
    support_count = _active_support_count(sample)
    support_content_tokens, support_content_sizes, support_style_tokens, support_style_sizes = _trimmed_support_data(sample)

    bundle_name = (
        f"{split}_{dataset_index:03d}_{Path(sample['filename']).stem}"
        f"_in_{sample['target_pack']}"
        f"_{support_count}shot"
    )
    result = save_prediction_bundle(
        output_dir=output_dir,
        bundle_name=bundle_name,
        content_tokens=sample["content_ref"],
        content_size=IMAGE_SIZE,
        support_content_tokens=support_content_tokens,
        support_content_sizes=support_content_sizes,
        support_style_tokens=support_style_tokens,
        support_style_sizes=support_style_sizes,
        prediction_tokens=prediction,
        prediction_size=IMAGE_SIZE,
        truth_tokens=sample["target"],
        truth_size=IMAGE_SIZE,
        metadata={
            "split": split,
            "index": dataset_index,
            "filename": sample["filename"],
            "content_filename": sample["content_filename"],
            "support_content_filenames": sample["support_content_filenames"],
            "support_style_filenames": sample["support_style_filenames"],
            "target_filename": sample["target_filename"],
            "content_pack": sample["content_pack"],
            "style_pack": sample["style_pack"],
            "target_pack": sample["target_pack"],
            "style": sample.get("style", "unspecified"),
            "checkpoint_path": str(checkpoint_path.resolve()),
        },
    )

    print(f"Evaluated split={split} index={dataset_index} filename={sample['filename']} pack={sample['target_pack']}")
    print(f"Support pack: {sample['style_pack']} using {sample['support_style_filenames']}")
    print(f"Saved original texture to {result['original_path']}")
    print(f"Saved support pairs to {result['support_path']}")
    print(f"Saved generated texture to {result['produced_path']}")
    print(f"Saved source of truth to {result['truth_path']}")
    print(f"Saved side-by-side comparison to {result['comparison_path']}")

    metrics = result["metrics"]
    if metrics is not None:
        print(
            "Metrics: "
            f"pixel_accuracy={metrics['pixel_accuracy']:.4f} "
            f"rgb_mae={metrics['rgb_mae']:.2f} "
            f"exact_match={metrics['exact_match']}"
        )

    return result["bundle_dir"]


def run(
    checkpoint_dir: str | Path = CHECKPOINTS_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    split: str = "val",
    mode: str = "matrix",
    index: int = 0,
    filename: str | None = None,
    target_pack: str | None = None,
):
    """Evaluate either the fixed validation matrix or one dataset example."""
    if mode not in {"matrix", "single"}:
        raise ValueError(f"Unsupported evaluation mode {mode!r}")

    if mode == "single":
        return _run_single(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            split=split,
            index=index,
            filename=filename,
            target_pack=target_pack,
        )

    dataset = TextureDataset(split=split)
    if len(dataset) == 0:
        raise ValueError(f"Dataset split {split!r} is empty. Run preprocessing first.")

    model, checkpoint_path = load_model(checkpoint_dir)
    summary = write_validation_matrix(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
    )
    print(f"Saved validation matrix for split={split} to {Path(output_dir).resolve()}")
    print(
        "Overall metrics: "
        f"examples={summary['overall']['examples']} "
        f"mean_pixel_accuracy={summary['overall']['mean_pixel_accuracy']:.4f} "
        f"mean_rgb_mae={summary['overall']['mean_rgb_mae']:.2f} "
        f"exact_match_count={summary['overall']['exact_match_count']}"
        if summary["overall"]["mean_pixel_accuracy"] is not None
        else "No metrics were produced."
    )
    return output_dir
