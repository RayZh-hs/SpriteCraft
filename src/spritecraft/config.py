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
PACK_REPORT_PATH = PROCESSED_DIR / "pack_report.json"

IMAGE_SIZE = 32
NUM_TIMESTEPS = 40
MIN_SHARED_PACKS = 5
MIN_SUPPORT_EXEMPLARS = 3
MAX_SUPPORT_EXEMPLARS = 6
VALIDATION_MATRIX_EXAMPLES_PER_PACK = 4

# Per-pack dataset directories
PACK_DATASET_DIR = PROCESSED_DIR / "packs"

def pack_dataset_path(pack_id: str) -> pathlib.Path:
    return PACK_DATASET_DIR / pack_id / "dataset.npz"

def pack_pair_index_path(pack_id: str) -> pathlib.Path:
    return PACK_DATASET_DIR / pack_id / "pair_index.json"

def pack_checkpoint_dir(checkpoint_dir: pathlib.Path | str, pack_id: str) -> pathlib.Path:
    return pathlib.Path(checkpoint_dir) / pack_id

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
