#!/usr/bin/env python3
"""Export checkpointed diffusion training curves for the report.

Figure 4.4 should track the diffusion branch directly. Routed validation loss
flattens on hard packs once recolor fallback dominates, which hides continued
diffusion difficulty.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PACK_IDS = (
    "Ashen_16x",
    "Bare_Bones_1_21_11",
    "Cartoon_Texture_Pack",
    "Chibli_64x_Freepack",
)


def _load_training_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_snapshot_dat(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    interval: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("step train_loss\n")
        for row in rows:
            step = int(row["step"])
            if step % interval != 0:
                continue
            handle.write(f"{step} {float(row['loss']):.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/run33"),
        help="Run directory containing per-pack training_metrics.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/data"),
        help="Directory for generated .dat files.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=500,
        help="Step interval to export.",
    )
    parser.add_argument(
        "--pack",
        action="append",
        default=None,
        help="Specific pack to export. May be repeated.",
    )
    args = parser.parse_args()

    pack_ids = tuple(args.pack) if args.pack else PACK_IDS
    for pack_id in pack_ids:
        metrics_path = args.checkpoint_root / pack_id / "training_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing training metrics for {pack_id}: {metrics_path}")
        rows = _load_training_rows(metrics_path)
        output_path = args.output_dir / f"{pack_id}_training_snapshot.dat"
        _write_snapshot_dat(rows, output_path, interval=args.interval)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
