"""Preprocessing pipeline."""

import json
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
    PAIR_INDEX_PATH,
    PALETTE_PATH,
    PALETTE_SIZE,
    PROCESSED_DIR,
    RAW_PACKS_DIR,
    VALIDATION_FILENAMES,
)

FULL_CUBE_FACES = frozenset({"down", "up", "north", "south", "west", "east"})


def detect_base_pack_id(pack_ids: list[str]) -> str:
    """Pick the pack that should act as the vanilla/original anchor."""
    prioritized_matches = [
        pack_id
        for pack_id in pack_ids
        if "vanilla" in pack_id.lower()
    ]
    if len(prioritized_matches) == 1:
        return prioritized_matches[0]
    if len(prioritized_matches) > 1:
        raise ValueError(
            f"Found multiple vanilla-like packs {prioritized_matches}; cannot infer a unique base pack."
        )

    raise ValueError(
        f"Could not infer a base pack from {pack_ids}. Include exactly one pack name containing 'vanilla'."
    )


def _normalize_asset_reference(ref: str, asset_kind: str, suffix: str) -> tuple[str, Path]:
    """Normalize Minecraft-style asset refs like `minecraft:block/stone`."""
    namespace, relative_path = ref.split(":", 1) if ":" in ref else ("minecraft", ref)
    normalized_suffix = suffix if relative_path.endswith(suffix) else f"{relative_path}{suffix}"
    return namespace, Path("assets") / namespace / asset_kind / normalized_suffix


def _iter_model_references(entry: Any):
    """Yield model references from blockstate `variants` / `multipart` entries."""
    if isinstance(entry, dict):
        model_name = entry.get("model")
        if isinstance(model_name, str):
            yield model_name

        apply_entry = entry.get("apply")
        if apply_entry is not None:
            yield from _iter_model_references(apply_entry)
    elif isinstance(entry, list):
        for item in entry:
            yield from _iter_model_references(item)


def _resolve_model(
    pack_dir: Path,
    model_ref: str,
    cache: dict[str, dict[str, Any]],
    active_stack: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve a block model through its parent chain."""
    if model_ref in cache:
        return cache[model_ref]

    if active_stack is None:
        active_stack = set()
    if model_ref in active_stack:
        raise ValueError(f"Detected cyclic model inheritance at {model_ref}")

    active_stack.add(model_ref)
    _namespace, model_path = _normalize_asset_reference(model_ref, "models", ".json")
    model_file = pack_dir / model_path
    with open(model_file, encoding="utf-8") as file_obj:
        model_data = json.load(file_obj)

    parent_ref = model_data.get("parent")
    parent_model: dict[str, Any] | None = None
    if isinstance(parent_ref, str):
        parent_model = _resolve_model(pack_dir, parent_ref, cache, active_stack)

    textures = dict(parent_model["textures"]) if parent_model is not None else {}
    textures.update(model_data.get("textures", {}))

    elements = model_data.get("elements")
    if elements is None and parent_model is not None:
        elements = parent_model["elements"]

    resolved_model = {
        "textures": textures,
        "elements": elements or [],
    }
    cache[model_ref] = resolved_model
    active_stack.remove(model_ref)
    return resolved_model


def _is_full_cube_model(model_data: dict[str, Any]) -> bool:
    """Return True when the resolved model includes a full 16x16x16 cube shell."""
    for element in model_data.get("elements", []):
        if element.get("from") != [0, 0, 0] or element.get("to") != [16, 16, 16]:
            continue

        faces = element.get("faces", {})
        if FULL_CUBE_FACES.issubset(faces):
            return True

    return False


def _resolve_texture_reference(texture_ref: str, textures: dict[str, str]) -> str | None:
    """Resolve `#aliases` in a model texture reference."""
    visited: set[str] = set()
    resolved_ref = texture_ref
    while resolved_ref.startswith("#"):
        if resolved_ref in visited:
            return None
        visited.add(resolved_ref)

        alias = resolved_ref[1:]
        next_ref = textures.get(alias)
        if not isinstance(next_ref, str):
            return None
        resolved_ref = next_ref

    return resolved_ref


def _collect_model_face_textures(model_data: dict[str, Any]) -> set[str]:
    """Collect block texture filenames used by model faces."""
    allowed_filenames: set[str] = set()
    textures = model_data.get("textures", {})

    for element in model_data.get("elements", []):
        for face in element.get("faces", {}).values():
            texture_ref = face.get("texture")
            if not isinstance(texture_ref, str):
                continue

            resolved_ref = _resolve_texture_reference(texture_ref, textures)
            if resolved_ref is None:
                continue

            namespace, texture_path = _normalize_asset_reference(resolved_ref, "textures", ".png")
            if namespace != "minecraft" or texture_path.parent.name != "block":
                continue

            allowed_filenames.add(texture_path.name)

    return allowed_filenames


def collect_allowed_block_texture_filenames(pack_dir: Path) -> set[str]:
    """Collect texture filenames used by full-cube Minecraft block models."""
    blockstates_dir = pack_dir / "assets" / "minecraft" / "blockstates"
    if not blockstates_dir.exists():
        raise FileNotFoundError(f"Missing blockstates directory in base pack: {blockstates_dir}")

    model_cache: dict[str, dict[str, Any]] = {}
    allowed_filenames: set[str] = set()
    for blockstate_path in sorted(blockstates_dir.glob("*.json")):
        with open(blockstate_path, encoding="utf-8") as file_obj:
            blockstate_data = json.load(file_obj)

        model_refs = set()
        variants = blockstate_data.get("variants")
        if variants is not None:
            model_refs.update(model_ref for entry in variants.values() for model_ref in _iter_model_references(entry))

        multipart = blockstate_data.get("multipart")
        if multipart is not None:
            model_refs.update(model_ref for entry in multipart for model_ref in _iter_model_references(entry))

        for model_ref in model_refs:
            resolved_model = _resolve_model(pack_dir, model_ref, model_cache)
            if not _is_full_cube_model(resolved_model):
                continue
            allowed_filenames.update(_collect_model_face_textures(resolved_model))

    return allowed_filenames


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


def collect_images(pack_dir: Path, allowed_filenames: set[str] | None = None) -> dict[str, Image.Image]:
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
        if allowed_filenames is not None and img_path.name not in allowed_filenames:
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
    base_pack_id = detect_base_pack_id([get_pack_id(pack_path) for pack_path in pack_files])

    allowed_texture_filenames: set[str] | None = None
    for pack_path in pack_files:
        if get_pack_id(pack_path) != base_pack_id:
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "pack"
            extract_pack(pack_path, extract_dir)
            allowed_texture_filenames = collect_allowed_block_texture_filenames(extract_dir)
        break

    if not allowed_texture_filenames:
        raise ValueError(f"Could not derive any allowed block textures from base pack {base_pack_id}")
    print(f"Allowed full-cube block textures: {len(allowed_texture_filenames)}")

    # Extract and collect images per pack
    all_pack_images: dict[str, dict[str, Image.Image]] = {}
    for pack_path in pack_files:
        pack_id = get_pack_id(pack_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "pack"
            extract_pack(pack_path, extract_dir)
            images = collect_images(extract_dir, allowed_texture_filenames)
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
        if filename not in VALIDATION_FILENAMES
    }
    val_pair_index = {
        filename: pairs
        for filename, pairs in pair_index.items()
        if filename in VALIDATION_FILENAMES
    }

    # Save pair index
    pair_data = {
        "train": train_pair_index,
        "val": val_pair_index,
        "filenames_per_pack": filenames_per_pack,
        "base_pack_id": base_pack_id,
        "validation_filenames": sorted(VALIDATION_FILENAMES),
    }
    with open(PAIR_INDEX_PATH, "w") as f:
        json.dump(pair_data, f, indent=2)
    print(f"Pair index saved to {PAIR_INDEX_PATH}")
    print(f"  Training pairs: {len(train_pair_index)}")
    print(f"  Validation pairs: {len(val_pair_index)}")
    print("Preprocessing complete.")
