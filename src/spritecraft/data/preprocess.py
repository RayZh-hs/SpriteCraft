"""Preprocessing pipeline."""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from spritecraft.config import (
    MANIFEST_JSON_PATH,
    DATASET_PATH,
    IMAGE_SIZE,
    MAX_SUPPORT_EXEMPLARS,
    MIN_SHARED_PACKS,
    MIN_SUPPORT_EXEMPLARS,
    PACK_REPORT_PATH,
    PAIR_INDEX_PATH,
    PALETTE_PATH,
    PALETTE_SIZE,
    PROCESSED_DIR,
    RAW_PACKS_DIR,
    VALIDATION_FILENAMES,
    VALIDATION_MATRIX_EXAMPLES_PER_PACK,
)
from spritecraft.data.support_index import (
    compute_texture_descriptor,
    infer_texture_family,
    rank_support_candidates,
)

FULL_CUBE_FACES = frozenset({"down", "up", "north", "south", "west", "east"})
SUPPORTED_PACK_ROLES = frozenset({"base", "train", "defer"})
SELECTED_PACK_ROLES = frozenset({"base", "train"})


@dataclass(frozen=True)
class PackSpec:
    """Resolved metadata for one pack archive."""

    pack_id: str
    archive_name: str
    archive_path: Path
    role: str
    style: str
    selected: bool


def detect_base_pack_id(pack_ids: list[str]) -> str:
    """Pick the pack that should act as the vanilla/original anchor."""
    prioritized_matches = [pack_id for pack_id in pack_ids if "vanilla" in pack_id.lower()]
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


def _strip_disabled_suffix(filename: str) -> str:
    if filename.endswith(".disabled"):
        return filename[: -len(".disabled")]
    return filename


def _is_supported_pack_archive(path: Path) -> bool:
    archive_name = _strip_disabled_suffix(path.name).lower()
    return archive_name.endswith(".zip") or archive_name.endswith(".jar")


def extract_pack(pack_path: Path, extract_dir: Path) -> Path:
    """Extract a .zip or .jar pack to a directory."""
    if not _is_supported_pack_archive(pack_path):
        raise ValueError(f"Unsupported pack format: {pack_path.name}")

    with zipfile.ZipFile(pack_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def get_pack_id(pack_path: Path) -> str:
    """Derive a pack ID from the archive filename."""
    normalized_name = _strip_disabled_suffix(pack_path.name)
    return Path(normalized_name).stem.replace(" ", "_").replace(".", "_")


def should_keep_image(img_path: Path, img: Image.Image) -> bool:
    """Return True if the image passes all filter rules."""
    if img_path.suffix.lower() != ".png":
        return False

    width, height = img.size
    if width != height:
        return False
    if width not in {16, 32, 64}:
        return False
    if img_path.with_suffix(".png.mcmeta").exists():
        return False
    if "_overlay" in img_path.stem.lower():
        return False
    if img.mode not in {"RGBA", "RGB", "P", "L"}:
        return False

    return True


def preprocess_image(img: Image.Image) -> Image.Image:
    """Resize and handle alpha for a single image."""
    width, _height = img.size

    if img.mode == "P":
        if "transparency" in img.info:
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    if img.mode == "RGBA":
        magenta = Image.new("RGBA", img.size, (255, 0, 255, 255))
        img = Image.alpha_composite(magenta, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if width == 16:
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    elif width == 64:
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    return img


def collect_images(
    pack_dir: Path,
    allowed_filenames: set[str] | None = None,
) -> tuple[dict[str, Image.Image], int]:
    """Collect all valid block textures from an extracted pack."""
    block_dir = pack_dir / "assets" / "minecraft" / "textures" / "block"
    if not block_dir.exists():
        return {}, 0

    images: dict[str, Image.Image] = {}
    block_texture_paths = sorted(block_dir.rglob("*.png"))
    total_textures = len(block_texture_paths)
    
    for img_path in block_texture_paths:
        try:
            img = Image.open(img_path)
            img.load()
        except Exception:
            continue

        if not should_keep_image(img_path, img):
            continue

        # Determine the canonical filename for this texture
        if img_path.parent == block_dir:
            # Flat structure: block/foo.png -> foo.png
            canonical_name = img_path.name
        else:
            # Nested structure: block/foo/1.png -> foo.png (use parent dir name)
            canonical_name = img_path.parent.name + ".png"
        
        if allowed_filenames is not None and canonical_name not in allowed_filenames:
            continue

        # Only keep the first matching texture for each canonical name
        if canonical_name not in images:
            images[canonical_name] = preprocess_image(img)

    return images, total_textures


def build_palette(all_pixels: np.ndarray) -> np.ndarray:
    """Build a 256-color palette with MiniBatchKMeans."""
    kmeans = MiniBatchKMeans(
        n_clusters=PALETTE_SIZE,
        random_state=42,
        batch_size=4096,
        n_init=3,
    )
    kmeans.fit(all_pixels)
    return kmeans.cluster_centers_.astype(np.uint8)


def quantize_image(img: Image.Image, palette: np.ndarray) -> np.ndarray:
    """Map an RGB image to palette indices."""
    arr = np.array(img, dtype=np.float32).reshape(-1, 3)
    dists = np.linalg.norm(arr[:, None, :] - palette[None, :, :], axis=2)
    return np.argmin(dists, axis=1).astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)


def _load_manifest(
    packs_dir: Path,
    manifest_path: Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if manifest_path is not None:
        candidate_paths = [Path(manifest_path)]
    else:
        candidate_paths = [packs_dir / "manifest.json"]

    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        with candidate_path.open(encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)

        if manifest is None:
            continue
        return manifest, candidate_path

    return None, None


def _discover_pack_archives(packs_dir: Path) -> dict[str, Path]:
    pack_paths = sorted(path for path in packs_dir.iterdir() if path.is_file() and _is_supported_pack_archive(path))
    discovered_packs: dict[str, Path] = {}
    for pack_path in pack_paths:
        pack_id = get_pack_id(pack_path)
        if pack_id in discovered_packs:
            raise ValueError(f"Duplicate pack id {pack_id!r} for {discovered_packs[pack_id]} and {pack_path}")
        discovered_packs[pack_id] = pack_path
    return discovered_packs


def _resolve_manifest_pack_specs(
    packs_dir: Path,
    discovered_packs: dict[str, Path],
    manifest: dict[str, Any],
) -> tuple[list[PackSpec], str]:
    manifest_entries = manifest.get("packs")
    if not isinstance(manifest_entries, list) or not manifest_entries:
        raise ValueError("Manifest must define a non-empty 'packs' list.")

    pack_specs_by_id: dict[str, PackSpec] = {}
    base_pack_id: str | None = None
    for entry in manifest_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entries must be mappings, got {entry!r}")

        pack_id = str(entry.get("id", "")).strip()
        archive_name = str(entry.get("archive", "")).strip()
        role = str(entry.get("role", "")).strip()
        style = str(entry.get("style", "")).strip() or "unspecified"
        if not pack_id:
            raise ValueError(f"Manifest entry missing pack id: {entry!r}")
        if role not in SUPPORTED_PACK_ROLES:
            raise ValueError(f"Manifest role for {pack_id!r} must be one of {sorted(SUPPORTED_PACK_ROLES)}")

        if archive_name:
            archive_path = packs_dir / archive_name
            if not archive_path.exists():
                raise FileNotFoundError(f"Manifest references missing archive {archive_path}")
            if get_pack_id(archive_path) != pack_id:
                raise ValueError(
                    f"Manifest entry {pack_id!r} does not match archive-derived id {get_pack_id(archive_path)!r}"
                )
        else:
            archive_path = discovered_packs.get(pack_id)
            if archive_path is None:
                raise FileNotFoundError(
                    f"Manifest entry {pack_id!r} has no archive field and no matching archive in {packs_dir}"
                )
            archive_name = archive_path.name

        selected = role in SELECTED_PACK_ROLES
        pack_specs_by_id[pack_id] = PackSpec(
            pack_id=pack_id,
            archive_name=archive_name,
            archive_path=archive_path,
            role=role,
            style=style,
            selected=selected,
        )
        if role == "base":
            if base_pack_id is not None and base_pack_id != pack_id:
                raise ValueError(f"Manifest defines multiple base packs: {base_pack_id!r} and {pack_id!r}")
            base_pack_id = pack_id

    if base_pack_id is None:
        raise ValueError("Manifest must define exactly one pack with role 'base'.")

    for pack_id, archive_path in discovered_packs.items():
        if pack_id in pack_specs_by_id:
            continue
        pack_specs_by_id[pack_id] = PackSpec(
            pack_id=pack_id,
            archive_name=archive_path.name,
            archive_path=archive_path,
            role="defer" if archive_path.name.endswith(".disabled") else "defer",
            style="unspecified",
            selected=False,
        )

    return sorted(pack_specs_by_id.values(), key=lambda spec: (not spec.selected, spec.role, spec.pack_id)), base_pack_id


def _resolve_fallback_pack_specs(discovered_packs: dict[str, Path]) -> tuple[list[PackSpec], str]:
    active_pack_ids = [
        pack_id
        for pack_id, pack_path in discovered_packs.items()
        if not pack_path.name.endswith(".disabled")
    ]
    if not active_pack_ids:
        raise FileNotFoundError("No active .zip or .jar packs found.")

    base_pack_id = detect_base_pack_id(sorted(active_pack_ids))
    pack_specs = []
    for pack_id, archive_path in discovered_packs.items():
        selected = not archive_path.name.endswith(".disabled")
        role = "base" if pack_id == base_pack_id else ("train" if selected else "defer")
        pack_specs.append(
            PackSpec(
                pack_id=pack_id,
                archive_name=archive_path.name,
                archive_path=archive_path,
                role=role,
                style="unspecified",
                selected=selected,
            )
        )

    pack_specs.sort(key=lambda spec: (not spec.selected, spec.role, spec.pack_id))
    return pack_specs, base_pack_id


def _build_support_rankings(
    base_images: dict[str, Image.Image],
    filenames_per_pack: dict[str, list[str]],
    split_pairs: dict[str, list[list[str | int]]],
    base_pack_id: str,
) -> dict[str, dict[str, list[str]]]:
    descriptors = {
        filename: compute_texture_descriptor(np.asarray(image, dtype=np.uint8))
        for filename, image in base_images.items()
    }
    base_filenames = set(filenames_per_pack.get(base_pack_id, []))
    support_pool_by_pack = {
        pack_id: sorted((base_filenames & set(filenames)) - VALIDATION_FILENAMES)
        for pack_id, filenames in filenames_per_pack.items()
        if pack_id != base_pack_id
    }

    support_rankings: dict[str, dict[str, list[str]]] = {}
    for filename, pairs in split_pairs.items():
        for pack_id, _array_idx in pairs:
            if not isinstance(pack_id, str):
                continue
            if pack_id == base_pack_id:
                continue
            candidate_filenames = [
                candidate_filename
                for candidate_filename in support_pool_by_pack.get(pack_id, [])
                if candidate_filename != filename
            ]
            ranked_candidates = rank_support_candidates(filename, candidate_filenames, descriptors)
            if ranked_candidates:
                support_rankings.setdefault(pack_id, {})[filename] = ranked_candidates

    return support_rankings


def _build_deterministic_supports(
    support_rankings: dict[str, dict[str, list[str]]],
    support_count: int,
) -> dict[str, dict[str, list[str]]]:
    deterministic_supports: dict[str, dict[str, list[str]]] = {}
    for pack_id, rankings_for_pack in support_rankings.items():
        for filename, ranked_candidates in rankings_for_pack.items():
            selected_supports = ranked_candidates[: min(support_count, len(ranked_candidates))]
            if selected_supports:
                deterministic_supports.setdefault(pack_id, {})[filename] = selected_supports
    return deterministic_supports


def _select_validation_entries(
    entries: list[dict[str, str | list[str]]],
    limit: int,
) -> list[dict[str, str | list[str]]]:
    if len(entries) <= limit:
        return sorted(entries, key=lambda entry: str(entry["filename"]))

    selected_entries: list[dict[str, str | list[str]]] = []
    used_families: set[str] = set()
    for entry in sorted(entries, key=lambda item: str(item["filename"])):
        family = infer_texture_family(str(entry["filename"]))
        if family in used_families:
            continue
        selected_entries.append(entry)
        used_families.add(family)
        if len(selected_entries) == limit:
            return selected_entries

    for entry in sorted(entries, key=lambda item: str(item["filename"])):
        if entry in selected_entries:
            continue
        selected_entries.append(entry)
        if len(selected_entries) == limit:
            break
    return selected_entries


def _build_validation_matrix(
    val_pair_index: dict[str, list[list[str | int]]],
    deterministic_supports: dict[str, dict[str, list[str]]],
    pack_specs: list[PackSpec],
    base_pack_id: str,
) -> list[dict[str, str | list[str]]]:
    spec_by_pack_id = {spec.pack_id: spec for spec in pack_specs}
    entries_by_pack: dict[str, list[dict[str, str | list[str]]]] = defaultdict(list)

    for filename, pairs in val_pair_index.items():
        for pack_id, _array_idx in pairs:
            if not isinstance(pack_id, str):
                continue
            if pack_id == base_pack_id:
                continue
            support_filenames = deterministic_supports.get(pack_id, {}).get(filename)
            if not support_filenames:
                continue

            spec = spec_by_pack_id[pack_id]
            entries_by_pack[pack_id].append(
                {
                    "split": "val",
                    "filename": filename,
                    "target_pack": pack_id,
                    "style": spec.style,
                    "support_filenames": support_filenames,
                }
            )

    validation_matrix: list[dict[str, str | list[str]]] = []
    for pack_id in sorted(entries_by_pack):
        selected_entries = _select_validation_entries(
            entries_by_pack[pack_id],
            limit=VALIDATION_MATRIX_EXAMPLES_PER_PACK,
        )
        validation_matrix.extend(selected_entries)

    return validation_matrix


def run(
    packs_dir: str | Path = RAW_PACKS_DIR,
    manifest_path: str | Path | None = None,
    min_shared_packs: int = MIN_SHARED_PACKS,
):
    """Run full preprocessing pipeline."""
    packs_dir = Path(packs_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    discovered_packs = _discover_pack_archives(packs_dir)
    if not discovered_packs:
        raise FileNotFoundError(f"No supported pack archives found in {packs_dir}")
    
    # If the configured manifest_path has a file, use it as the default
    if manifest_path is None and MANIFEST_JSON_PATH.exists():
        manifest_path = MANIFEST_JSON_PATH

    manifest, resolved_manifest_path = _load_manifest(
        packs_dir=packs_dir,
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
    )
    if manifest is not None:
        pack_specs, base_pack_id = _resolve_manifest_pack_specs(packs_dir, discovered_packs, manifest)
    else:
        pack_specs, base_pack_id = _resolve_fallback_pack_specs(discovered_packs)

    selected_pack_specs = [spec for spec in pack_specs if spec.selected]
    if base_pack_id not in {spec.pack_id for spec in selected_pack_specs}:
        raise ValueError(f"Base pack {base_pack_id!r} is not selected.")

    print(f"Found {len(pack_specs)} pack archive(s): {[spec.archive_name for spec in pack_specs]}")
    if resolved_manifest_path is not None:
        print(f"Using pack manifest {resolved_manifest_path}")
    else:
        print("No manifest found; using directory-scan selection.")
    print(f"Selected {len(selected_pack_specs)} pack(s): {[spec.pack_id for spec in selected_pack_specs]}")
    print(f"Base pack: {base_pack_id}")
    print(f"Minimum shared packs threshold: {min_shared_packs}")

    base_spec = next(spec for spec in pack_specs if spec.pack_id == base_pack_id)
    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir) / "pack"
        extract_pack(base_spec.archive_path, extract_dir)
        allowed_texture_filenames = collect_allowed_block_texture_filenames(extract_dir)

    if not allowed_texture_filenames:
        raise ValueError(f"Could not derive any allowed block textures from base pack {base_pack_id}")
    print(f"Allowed full-cube block textures: {len(allowed_texture_filenames)}")

    all_pack_images: dict[str, dict[str, Image.Image]] = {}
    pack_report: dict[str, Any] = {
        "base_pack_id": base_pack_id,
        "manifest_path": str(resolved_manifest_path.resolve()) if resolved_manifest_path is not None else None,
        "selected_pack_ids": [spec.pack_id for spec in selected_pack_specs],
        "packs": {},
    }

    for spec in pack_specs:
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "pack"
            extract_pack(spec.archive_path, extract_dir)
            images, total_block_textures = collect_images(extract_dir, allowed_texture_filenames)

        shared_full_cube_count = len(set(images) & allowed_texture_filenames)
        validation_overlap_count = len(set(images) & VALIDATION_FILENAMES)
        pack_report["packs"][spec.pack_id] = {
            "pack_id": spec.pack_id,
            "archive_name": spec.archive_name,
            "role": spec.role,
            "style": spec.style,
            "selected": spec.selected,
            "total_block_textures": total_block_textures,
            "valid_textures": len(images),
            "shared_full_cube_texture_count": shared_full_cube_count,
            "validation_overlap_count": validation_overlap_count,
        }
        print(
            "  "
            f"{spec.pack_id}: role={spec.role} selected={spec.selected} "
            f"valid={len(images)} shared={shared_full_cube_count} val_overlap={validation_overlap_count}"
        )
        if spec.selected:
            all_pack_images[spec.pack_id] = images

    with PACK_REPORT_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(pack_report, file_obj, indent=2)
    print(f"Pack report saved to {PACK_REPORT_PATH}")

    all_pixels = [
        np.asarray(image, dtype=np.uint8).reshape(-1, 3)
        for images in all_pack_images.values()
        for image in images.values()
    ]
    if not all_pixels:
        raise ValueError("No selected pack images were collected. Update the manifest or pack directory.")

    all_pixel_array = np.concatenate(all_pixels, axis=0)
    print(f"Total pixels for palette: {all_pixel_array.shape[0]:,}")
    palette = build_palette(all_pixel_array)
    np.save(PALETTE_PATH, palette)
    print(f"Palette saved to {PALETTE_PATH}")

    pack_arrays: dict[str, dict[str, np.ndarray]] = {}
    for pack_id, images in all_pack_images.items():
        pack_arrays[pack_id] = {
            filename: quantize_image(image, palette)
            for filename, image in images.items()
        }

    flat_arrays: dict[str, Any] = {}
    filenames_per_pack: dict[str, list[str]] = {}
    for pack_id, arr_dict in pack_arrays.items():
        filenames = sorted(arr_dict)
        filenames_per_pack[pack_id] = filenames
        stack = np.stack([arr_dict[filename] for filename in filenames], axis=0)
        flat_arrays[pack_id] = stack
        print(f"  {pack_id}: quantized shape {stack.shape}")

    np.savez(DATASET_PATH, **flat_arrays)
    print(f"Dataset saved to {DATASET_PATH}")

    raw_pair_index: dict[str, list[list[str | int]]] = defaultdict(list)
    for pack_id, filenames in filenames_per_pack.items():
        for array_idx, filename in enumerate(filenames):
            raw_pair_index[filename].append([pack_id, array_idx])

    filtered_pair_index = {
        filename: sorted(pairs, key=lambda pair: str(pair[0]))
        for filename, pairs in raw_pair_index.items()
        if len(pairs) >= min_shared_packs
    }
    train_pair_index = {
        filename: pairs
        for filename, pairs in filtered_pair_index.items()
        if filename not in VALIDATION_FILENAMES
    }
    val_pair_index = {
        filename: pairs
        for filename, pairs in filtered_pair_index.items()
        if filename in VALIDATION_FILENAMES
    }

    support_rankings = {
        "train": _build_support_rankings(
            base_images=all_pack_images[base_pack_id],
            filenames_per_pack=filenames_per_pack,
            split_pairs=train_pair_index,
            base_pack_id=base_pack_id,
        ),
        "val": _build_support_rankings(
            base_images=all_pack_images[base_pack_id],
            filenames_per_pack=filenames_per_pack,
            split_pairs=val_pair_index,
            base_pack_id=base_pack_id,
        ),
    }
    deterministic_supports = {
        "val": _build_deterministic_supports(
            support_rankings["val"],
            support_count=MIN_SUPPORT_EXEMPLARS,
        )
    }
    validation_matrix = _build_validation_matrix(
        val_pair_index=val_pair_index,
        deterministic_supports=deterministic_supports["val"],
        pack_specs=pack_specs,
        base_pack_id=base_pack_id,
    )

    pair_data = {
        "train": train_pair_index,
        "val": val_pair_index,
        "filenames_per_pack": filenames_per_pack,
        "base_pack_id": base_pack_id,
        "validation_filenames": sorted(VALIDATION_FILENAMES),
        "selected_pack_ids": [spec.pack_id for spec in selected_pack_specs],
        "pack_roles": {spec.pack_id: spec.role for spec in pack_specs},
        "pack_styles": {spec.pack_id: spec.style for spec in pack_specs},
        "min_shared_packs": min_shared_packs,
        "support_count_range": {
            "min": MIN_SUPPORT_EXEMPLARS,
            "max": MAX_SUPPORT_EXEMPLARS,
        },
        "support_rankings": support_rankings,
        "deterministic_supports": deterministic_supports,
        "validation_matrix": validation_matrix,
    }
    with PAIR_INDEX_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(pair_data, file_obj, indent=2)

    print(f"Pair index saved to {PAIR_INDEX_PATH}")
    print(f"  Training filenames: {len(train_pair_index)}")
    print(f"  Validation filenames: {len(val_pair_index)}")
    target_pack_ids: set[str] = set()
    for entry in validation_matrix:
        target_pack = entry.get("target_pack")
        if isinstance(target_pack, str):
            target_pack_ids.add(target_pack)
    print(
        "  Validation matrix entries: "
        f"{len(validation_matrix)} across {len(target_pack_ids)} pack(s)"
    )
    print("Preprocessing complete.")
