#!/usr/bin/env python3
"""Generate validation bundles for checkpoint directories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spritecraft.inference import generate
from spritecraft.config import pack_pair_index_path


def _discover_packs(checkpoint_root: Path) -> list[str]:
    packs = []
    for child in sorted(checkpoint_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "latest.pt").exists():
            packs.append(child.name)
    return packs


def _filenames_for_split(pack_id: str, split: str) -> list[str]:
    pair_index = json.loads(pack_pair_index_path(pack_id).read_text(encoding="utf-8"))
    key = "val_filenames" if split == "val" else "train_filenames"
    filenames = pair_index.get(key, [])
    if not isinstance(filenames, list):
        raise ValueError(f"Malformed {key} for pack {pack_id}")
    return filenames


def _tile(images: list[Image.Image], *, columns: int = 2, gap: int = 6) -> Image.Image:
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

    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        x = column * (cell_width + gap)
        y = row * (cell_height + gap)
        canvas.paste(image, (x, y))

    return canvas


def _write_summary(output_root: Path) -> Path:
    summary: dict[str, object] = {"packs": {}}
    overall_source_counts: Counter[str] = Counter()

    for pack_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        entries = []
        comparison_images: list[Image.Image] = []
        source_counts: Counter[str] = Counter()
        maes: list[float] = []
        accuracies: list[float] = []

        for bundle_dir in sorted(path for path in pack_dir.iterdir() if path.is_dir()):
            metadata_path = bundle_dir / "metadata.json"
            metrics_path = bundle_dir / "metrics.json"
            comparison_path = bundle_dir / "comparison.png"
            if not metadata_path.exists() or not metrics_path.exists():
                continue

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            model_source = str(metadata.get("model_source", metrics.get("model_source", "unknown")))
            source_counts[model_source] += 1
            overall_source_counts[model_source] += 1

            mae = metrics.get("mae")
            pixel_accuracy = metrics.get("pixel_accuracy")
            if isinstance(mae, (int, float)):
                maes.append(float(mae))
            if isinstance(pixel_accuracy, (int, float)):
                accuracies.append(float(pixel_accuracy))

            entries.append(
                {
                    "filename": metadata.get("filename"),
                    "model_source": model_source,
                    "model_score": metadata.get("model_score", metrics.get("model_score")),
                    "bundle_dir": str(bundle_dir.resolve()),
                    "source": str((bundle_dir / "original_tex.png").resolve()),
                    "selected": str((bundle_dir / "produced_tex.png").resolve()),
                    "ground_truth": str((bundle_dir / "source_of_truth.png").resolve()),
                    "comparison": str(comparison_path.resolve()),
                    "support": str((bundle_dir / "support.png").resolve()),
                    "metrics": metrics,
                }
            )

            if comparison_path.exists():
                with Image.open(comparison_path) as image:
                    comparison_images.append(image.convert("RGB").copy())

        if comparison_images:
            _tile(comparison_images).save(pack_dir / "summary.png")

        summary["packs"][pack_dir.name] = {
            "count": len(entries),
            "mean_mae": sum(maes) / len(maes) if maes else None,
            "mean_pixel_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
            "model_sources": dict(source_counts),
            "entries": entries,
        }

    summary["overall_model_sources"] = dict(overall_source_counts)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation bundles for checkpoint packs.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Checkpoint run directory, for example checkpoints/run33",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: output/<checkpoint_name>_<split>)",
    )
    parser.add_argument(
        "--split",
        choices=("val", "train"),
        default="val",
        help="Dataset split to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="Random seed forwarded to generation",
    )
    parser.add_argument(
        "--pack",
        action="append",
        default=None,
        help="Specific pack to evaluate; may be repeated",
    )
    args = parser.parse_args()

    checkpoint_root = args.checkpoint.resolve()
    output_root = args.output.resolve() if args.output is not None else PROJECT_ROOT / "output" / f"{checkpoint_root.name}_{args.split}"
    output_root.mkdir(parents=True, exist_ok=True)

    packs = args.pack if args.pack else _discover_packs(checkpoint_root)
    if not packs:
        raise ValueError(f"No complete checkpoints found in {checkpoint_root}")

    skipped: dict[str, str] = {}
    for pack_id in packs:
        try:
            textures = _filenames_for_split(pack_id, args.split)
            if not textures:
                skipped[pack_id] = f"empty {args.split} split"
                continue
            print(f"=== {pack_id} ({len(textures)} {args.split} textures) ===", flush=True)
            generate.run(
                pack_id=pack_id,
                checkpoint=checkpoint_root,
                output_dir=output_root,
                textures=textures,
                seed=args.seed,
            )
        except Exception as exc:  # pragma: no cover - convenience script
            skipped[pack_id] = str(exc)
            print(f"Skipping {pack_id}: {exc}", flush=True)

    summary_path = _write_summary(output_root)
    if skipped:
        skipped_path = output_root / "skipped.json"
        skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")
        print(f"Wrote skipped packs to {skipped_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
