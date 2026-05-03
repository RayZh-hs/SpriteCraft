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
