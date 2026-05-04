"""Shared configuration and constants."""

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_PACKS_DIR = DATA_DIR / "raw_packs"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"

PALETTE_PATH = PROCESSED_DIR / "palette.npy"
DATASET_PATH = PROCESSED_DIR / "dataset.npz"
PAIR_INDEX_PATH = PROCESSED_DIR / "pair_index.json"

IMAGE_SIZE = 32
PALETTE_SIZE = 256
MASK_TOKEN = 256
VOCAB_SIZE = 257  # 256 palette indices + 1 mask token
NUM_TIMESTEPS = 50
NUM_SUPPORT_EXEMPLARS = 3

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
])
