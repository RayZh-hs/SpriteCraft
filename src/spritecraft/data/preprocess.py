"""Preprocessing pipeline."""

import json
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from spritecraft.config import (
    DATASET_PATH,
    IMAGE_SIZE,
    MASK_TOKEN,
    PAIR_INDEX_PATH,
    PALETTE_PATH,
    PALETTE_SIZE,
    PROCESSED_DIR,
    RAW_PACKS_DIR,
)

VALIDATION_BLOCKS = frozenset([
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


def extract_pack(pack_path: Path, extract_dir: Path) -> Path:
    """Extract a .zip or .jar pack to a directory."""
    if pack_path.suffix.lower() in {".zip", ".jar"}:
        with zipfile.ZipFile(pack_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported pack format: {pack_path.suffix}")
    return extract_dir


def get_pack_id(pack_path: Path) -> str:
    """Derive a pack ID from the filename."""
    return pack_path.stem.replace(" ", "_").replace(".", "_")


def should_keep_image(img_path: Path, img: Image.Image) -> bool:
    """Return True if the image passes all filter rules."""
    # Must be .png
    if img_path.suffix.lower() != ".png":
        return False

    # Must be square
    w, h = img.size
    if w != h:
        return False

    # Width must be 16 or 32
    if w not in {16, 32}:
        return False

    # No .png.mcmeta neighbor
    if img_path.with_suffix(".png.mcmeta").exists():
        return False

    # No _overlay in filename
    if "_overlay" in img_path.stem.lower():
        return False

    # Mode must be RGBA, RGB, P (palette), or L (grayscale)
    if img.mode not in {"RGBA", "RGB", "P", "L"}:
        return False

    return True


def preprocess_image(img: Image.Image) -> Image.Image:
    """Resize and handle alpha for a single image."""
    w, h = img.size

    # Convert palette mode to RGB/RGBA first
    if img.mode == "P":
        if "transparency" in img.info:
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    # Handle alpha: composite onto magenta
    if img.mode == "RGBA":
        magenta = Image.new("RGBA", img.size, (255, 0, 255, 255))
        img = Image.alpha_composite(magenta, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize 16x to 32x
    if w == 16:
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)

    return img


def collect_images(pack_dir: Path) -> dict[str, Image.Image]:
    """Collect all valid block textures from an extracted pack."""
    block_dir = pack_dir / "assets" / "minecraft" / "textures" / "block"
    if not block_dir.exists():
        return {}

    images = {}
    for img_path in sorted(block_dir.glob("*.png")):
        try:
            img = Image.open(img_path)
            # Load early so we catch corrupted files
            img.load()
        except Exception:
            continue

        if not should_keep_image(img_path, img):
            continue

        img = preprocess_image(img)
        images[img_path.name] = img

    return images


def build_palette(all_pixels: np.ndarray) -> np.ndarray:
    """Build a 256-color palette with MiniBatchKMeans."""
    kmeans = MiniBatchKMeans(
        n_clusters=PALETTE_SIZE,
        random_state=42,
        batch_size=4096,
        n_init=3,
    )
    kmeans.fit(all_pixels)
    palette = kmeans.cluster_centers_.astype(np.uint8)
    return palette


def quantize_image(img: Image.Image, palette: np.ndarray) -> np.ndarray:
    """Map an RGB image to palette indices."""
    arr = np.array(img, dtype=np.float32).reshape(-1, 3)
    # Compute nearest centroid
    dists = np.linalg.norm(arr[:, None, :] - palette[None, :, :], axis=2)
    indices = np.argmin(dists, axis=1).astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
    return indices


def run(packs_dir: str | Path = RAW_PACKS_DIR):
    """Run full preprocessing pipeline."""
    packs_dir = Path(packs_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pack_files = sorted(packs_dir.glob("*.zip")) + sorted(packs_dir.glob("*.jar"))
    if not pack_files:
        raise FileNotFoundError(f"No .zip or .jar packs found in {packs_dir}")

    print(f"Found {len(pack_files)} pack(s): {[p.name for p in pack_files]}")

    # Extract and collect images per pack
    all_pack_images: dict[str, dict[str, Image.Image]] = {}
    for pack_path in pack_files:
        pack_id = get_pack_id(pack_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "pack"
            extract_pack(pack_path, extract_dir)
            images = collect_images(extract_dir)
            print(f"  {pack_id}: {len(images)} images")
            all_pack_images[pack_id] = images

    # Gather all pixels for palette
    all_pixels = []
    for images in all_pack_images.values():
        for img in images.values():
            all_pixels.append(np.array(img, dtype=np.uint8).reshape(-1, 3))
    all_pixels = np.concatenate(all_pixels, axis=0)
    print(f"Total pixels for palette: {all_pixels.shape[0]:,}")

    # Build palette
    palette = build_palette(all_pixels)
    np.save(PALETTE_PATH, palette)
    print(f"Palette saved to {PALETTE_PATH}")

    # Quantize all images and build flat arrays
    pack_arrays: dict[str, dict[str, np.ndarray]] = {}
    for pack_id, images in all_pack_images.items():
        pack_arrays[pack_id] = {}
        for filename, img in images.items():
            indices = quantize_image(img, palette)
            pack_arrays[pack_id][filename] = indices

    # Flatten into a single dataset array per pack
    # Typed as `Any` to bypass Pylance's strict dictionary unpacking check against numpy stubs
    flat_arrays: dict[str, Any] = {}
    filenames_per_pack: dict[str, list[str]] = {}
    for pack_id, arr_dict in pack_arrays.items():
        filenames = sorted(arr_dict.keys())
        filenames_per_pack[pack_id] = filenames
        stack = np.stack([arr_dict[f] for f in filenames], axis=0)
        flat_arrays[pack_id] = stack
        print(f"  {pack_id}: quantized shape {stack.shape}")

    # Save dataset.npz
    np.savez(DATASET_PATH, **flat_arrays)
    print(f"Dataset saved to {DATASET_PATH}")

    # Build pair index
    pair_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for pack_id, filenames in filenames_per_pack.items():
        for idx, filename in enumerate(filenames):
            pair_index[filename].append((pack_id, idx))

    # Keep only filenames present in 3+ packs
    pair_index = {
        filename: pairs
        for filename, pairs in pair_index.items()
        if len(pairs) >= 3
    }

    # Validation split: remove validation blocks from training pair index
    train_pair_index = {
        filename: pairs
        for filename, pairs in pair_index.items()
        if filename not in VALIDATION_BLOCKS
    }
    val_pair_index = {
        filename: pairs
        for filename, pairs in pair_index.items()
        if filename in VALIDATION_BLOCKS
    }

    # Save pair index
    pair_data = {
        "train": train_pair_index,
        "val": val_pair_index,
        "filenames_per_pack": filenames_per_pack,
    }
    with open(PAIR_INDEX_PATH, "w") as f:
        json.dump(pair_data, f, indent=2)
    print(f"Pair index saved to {PAIR_INDEX_PATH}")
    print(f"  Training pairs: {len(train_pair_index)}")
    print(f"  Validation pairs: {len(val_pair_index)}")
    print("Preprocessing complete.")
