#!/usr/bin/env python3
"""Generate compact texture-comparison figures for the report.

The script reads saved validation preview panels from a checkpoint directory
and writes the semantic report figures:

    report/figures/multi-pack-oak-planks.png
    report/figures/ashen-routing-examples.png

The preview panels are expected to use SpriteCraft's validation layout:
100x44 pixels for diffusion validation panels, with 32x32 content, output, and
target tiles at x offsets 0, 34, and 68 after a 12-pixel label row. The script
removes the selection border by trimming three pixels from each tile before
scaling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PACK_ROWS = (
    ("Ashen_16x", "Ashen"),
    ("Bare_Bones_1_21_11", "Bare Bones"),
    ("Cartoon_Texture_Pack", "Cartoon"),
    ("Chibli_64x_Freepack", "Chibli"),
)
ROUTING_ROWS = (
    ("stone", "stone"),
    ("oak_planks", "oak planks"),
    ("bookshelf", "bookshelf"),
    ("tnt_side", "tnt side"),
)
FONT_CANDIDATES = (
    Path("/usr/share/fonts/noto/NotoSans-Regular.ttf"),
    Path("/usr/share/fonts/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)
BACKGROUND = (250, 250, 248)
TEXT = (24, 24, 24)
BORDER = (205, 205, 198)


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _tile(panel_path: Path, which: str, trim: int) -> Image.Image:
    image = Image.open(panel_path).convert("RGB")
    if image.size == (100, 44):
        x_offset = {"content": 0, "output": 34, "target": 68}[which]
        y_offset = 12
    elif image.size == (96, 32):
        x_offset = {"content": 0, "output": 32, "target": 64}[which]
        y_offset = 0
    else:
        raise ValueError(f"Unexpected preview size {image.size}: {panel_path}")

    return image.crop((
        x_offset + trim,
        y_offset + trim,
        x_offset + 32 - trim,
        y_offset + 32 - trim,
    ))


def _draw_grid(
    *,
    rows: tuple[tuple[str, str], ...],
    columns: tuple[str, ...],
    get_tile,
    output_path: Path,
    label_width: int,
    scale: int,
    gap: int,
    trim: int,
) -> None:
    font = _load_font(13)
    tile_size = (32 - 2 * trim) * scale
    top = 28
    pad = 10
    width = label_width + len(columns) * tile_size + (len(columns) - 1) * gap + pad
    height = top + len(rows) * tile_size + (len(rows) - 1) * gap + pad
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    for column_index, column in enumerate(columns):
        x = label_width + column_index * (tile_size + gap)
        text_width, _ = _text_size(draw, column, font)
        draw.text((x + (tile_size - text_width) // 2, 8), column, fill=TEXT, font=font)

    for row_index, (row_key, row_label) in enumerate(rows):
        y = top + row_index * (tile_size + gap)
        _, label_height = _text_size(draw, row_label, font)
        draw.text((6, y + (tile_size - label_height) // 2 - 1), row_label, fill=TEXT, font=font)
        for column_index, column in enumerate(columns):
            x = label_width + column_index * (tile_size + gap)
            tile = get_tile(row_key, column).resize((tile_size, tile_size), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, y))
            draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline=BORDER, width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def generate_figures(
    *,
    checkpoint_root: Path,
    output_dir: Path,
    validation_step: int,
    pack_id: str,
    scale: int,
    trim: int,
    gap: int,
) -> None:
    step_dirname = f"step_{validation_step:06d}"

    _draw_grid(
        rows=PACK_ROWS,
        columns=("Content", "Selected", "Target"),
        get_tile=lambda pack, column: _tile(
            checkpoint_root / pack / "validation" / step_dirname / "oak_planks_selected.png",
            {"Content": "content", "Selected": "output", "Target": "target"}[column],
            trim,
        ),
        output_path=output_dir / "multi-pack-oak-planks.png",
        label_width=74,
        scale=scale,
        gap=gap,
        trim=trim,
    )

    def routing_tile(texture: str, column: str) -> Image.Image:
        branch = {
            "Content": "selected",
            "Diffusion": "diffusion",
            "Recolor": "recolor",
            "Selected": "selected",
            "Target": "selected",
        }[column]
        which = {
            "Content": "content",
            "Diffusion": "output",
            "Recolor": "output",
            "Selected": "output",
            "Target": "target",
        }[column]
        return _tile(
            checkpoint_root / pack_id / "validation" / step_dirname / f"{texture}_{branch}.png",
            which,
            trim,
        )

    _draw_grid(
        rows=ROUTING_ROWS,
        columns=("Content", "Diffusion", "Recolor", "Selected", "Target"),
        get_tile=routing_tile,
        output_path=output_dir / "ashen-routing-examples.png",
        label_width=86,
        scale=scale,
        gap=gap,
        trim=trim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/run39"),
        help="Checkpoint directory containing per-pack validation previews.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/figures"),
        help="Directory where generated report figures are written.",
    )
    parser.add_argument(
        "--validation-step",
        type=int,
        default=10_000,
        help="Validation step directory to read, e.g. 10000 for step_010000.",
    )
    parser.add_argument(
        "--pack-id",
        default="Ashen_16x",
        help="Pack used for routing examples.",
    )
    parser.add_argument("--scale", type=int, default=3, help="Nearest-neighbor scale for cropped tiles.")
    parser.add_argument("--trim", type=int, default=3, help="Pixels trimmed from each side of every tile.")
    parser.add_argument("--gap", type=int, default=8, help="Pixel gap between grid cells.")
    args = parser.parse_args()

    generate_figures(
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        validation_step=args.validation_step,
        pack_id=args.pack_id,
        scale=args.scale,
        trim=args.trim,
        gap=args.gap,
    )
    print(f"Wrote {args.output_dir / 'multi-pack-oak-planks.png'}")
    print(f"Wrote {args.output_dir / 'ashen-routing-examples.png'}")


if __name__ == "__main__":
    main()
