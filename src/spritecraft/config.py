"""Shared configuration and constants."""

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_PACKS_DIR = DATA_DIR / "raw_packs"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"

MANIFEST_JSON_PATH = DATA_DIR / "manifest.json"
PALETTE_PATH = PROCESSED_DIR / "palette.npy"
DATASET_PATH = PROCESSED_DIR / "dataset.npz"
PAIR_INDEX_PATH = PROCESSED_DIR / "pair_index.json"
PACK_REPORT_PATH = PROCESSED_DIR / "pack_report.json"

IMAGE_SIZE = 32
PALETTE_SIZE = 256
MASK_TOKEN = 256
VOCAB_SIZE = 257  # 256 palette indices + 1 mask token
NUM_TIMESTEPS = 50
MIN_SHARED_PACKS = 5
STYLE_DIM = 256

VALIDATION_FILENAMES = frozenset([
    "stone.png",
    "dirt.png",
    "cobblestone.png",
    "oak_planks.png",
    "spruce_planks.png",
    "sand.png",
    "gravel.png",
    "gold_ore.png",
    "iron_ore.png",
    "coal_ore.png",
    "oak_log.png",
    "oak_leaves.png",
    "glass.png",
    "diamond_ore.png",
    "farmland.png",
    "bricks.png",
    "tnt_side.png",
    "bookshelf.png",
    "mossy_cobblestone.png",
    "obsidian.png",
    "diamond.png",
    "iron_ingot.png",
    "stick.png",
    "apple.png",
    "bow.png",
])
