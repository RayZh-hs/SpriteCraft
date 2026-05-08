"""Preprocessing pipeline for per-pack RGB datasets."""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritecraft.config import (
    MANIFEST_JSON_PATH,
    IMAGE_SIZE,
    MIN_SHARED_PACKS,
    PACK_DATASET_DIR,
    PACK_REPORT_PATH,
    PROCESSED_DIR,
    RAW_PACKS_DIR,
    VALIDATION_FILENAMES,
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


def preprocess_image(img: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Resize image and extract RGB + alpha channels.
    
    Returns:
        rgb_image: PIL Image in RGB mode, resized to IMAGE_SIZE
        alpha: numpy array of shape (IMAGE_SIZE, IMAGE_SIZE) with values in [0, 1]
    """
    width, _height = img.size

    if img.mode == "P":
        if "transparency" in img.info:
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    if img.mode == "RGBA":
        rgb = img.convert("RGB")
        alpha = np.array(img)[:, :, 3].astype(np.float32) / 255.0
        if width == 16:
            rgb = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
            alpha = np.array(Image.fromarray((alpha * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)) / 255.0
        elif width == 64:
            rgb = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            alpha = np.array(Image.fromarray((alpha * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)) / 255.0
        return rgb, alpha
    else:
        rgb = img.convert("RGB")
        if width == 16:
            rgb = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
        elif width == 64:
            rgb = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        alpha = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        return rgb, alpha


def collect_images(
    pack_dir: Path,
    allowed_filenames: set[str] | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """Collect all valid block textures from an extracted pack.
    
    Returns:
        images: dict mapping filename to (rgb_array, alpha_array)
        total_textures: total number of textures scanned
    """
    block_dir = pack_dir / "assets" / "minecraft" / "textures" / "block"
    if not block_dir.exists():
        return {}, 0

    images: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
            canonical_name = img_path.name
        else:
            canonical_name = img_path.parent.name + ".png"
        
        if allowed_filenames is not None and canonical_name not in allowed_filenames:
            continue

        # Only keep the first matching texture for each canonical name
        if canonical_name not in images:
            rgb, alpha = preprocess_image(img)
            rgb_array = np.array(rgb, dtype=np.float32) / 255.0
            images[canonical_name] = (rgb_array, alpha)

    return images, total_textures


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


def run(
    packs_dir: str | Path = RAW_PACKS_DIR,
    manifest_path: str | Path | None = None,
    min_shared_packs: int = MIN_SHARED_PACKS,
):
    """Run full preprocessing pipeline, splitting data per pack."""
    packs_dir = Path(packs_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PACK_DATASET_DIR.mkdir(parents=True, exist_ok=True)

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

    all_pack_images: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
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

    # Build per-pack datasets
    base_images = all_pack_images[base_pack_id]
    base_filenames = set(base_images.keys())
    
    target_pack_specs = [spec for spec in selected_pack_specs if spec.pack_id != base_pack_id]
    
    for spec in target_pack_specs:
        pack_id = spec.pack_id
        target_images = all_pack_images[pack_id]
        target_filenames = set(target_images.keys())
        
        # Find shared filenames between base and target
        shared_filenames = sorted(base_filenames & target_filenames)
        
        if len(shared_filenames) < min_shared_packs:
            print(f"  {pack_id}: skipped (only {len(shared_filenames)} shared textures, need {min_shared_packs})")
            continue
        
        # Split into train/val
        train_filenames = [f for f in shared_filenames if f not in VALIDATION_FILENAMES]
        val_filenames = [f for f in shared_filenames if f in VALIDATION_FILENAMES]
        
        # Stack RGB and alpha arrays
        content_rgb_train = np.stack([base_images[f][0] for f in train_filenames], axis=0)
        content_alpha_train = np.stack([base_images[f][1] for f in train_filenames], axis=0)
        target_rgb_train = np.stack([target_images[f][0] for f in train_filenames], axis=0)
        target_alpha_train = np.stack([target_images[f][1] for f in train_filenames], axis=0)
        
        if val_filenames:
            content_rgb_val = np.stack([base_images[f][0] for f in val_filenames], axis=0)
            content_alpha_val = np.stack([base_images[f][1] for f in val_filenames], axis=0)
            target_rgb_val = np.stack([target_images[f][0] for f in val_filenames], axis=0)
            target_alpha_val = np.stack([target_images[f][1] for f in val_filenames], axis=0)
        else:
            content_rgb_val = np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            content_alpha_val = np.zeros((0, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            target_rgb_val = np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            target_alpha_val = np.zeros((0, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        
        # Also save all target pack textures for style reference sampling
        all_target_filenames = sorted(target_images.keys())
        all_target_rgb = np.stack([target_images[f][0] for f in all_target_filenames], axis=0)
        all_target_alpha = np.stack([target_images[f][1] for f in all_target_filenames], axis=0)
        
        # Save per-pack dataset
        pack_dir = PACK_DATASET_DIR / pack_id
        pack_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            pack_dir / "dataset.npz",
            content_rgb_train=content_rgb_train,
            content_alpha_train=content_alpha_train,
            target_rgb_train=target_rgb_train,
            target_alpha_train=target_alpha_train,
            content_rgb_val=content_rgb_val,
            content_alpha_val=content_alpha_val,
            target_rgb_val=target_rgb_val,
            target_alpha_val=target_alpha_val,
            all_target_rgb=all_target_rgb,
            all_target_alpha=all_target_alpha,
        )
        
        # Save pair index
        pair_index = {
            "pack_id": pack_id,
            "base_pack_id": base_pack_id,
            "train_filenames": train_filenames,
            "val_filenames": val_filenames,
            "all_target_filenames": all_target_filenames,
            "style": spec.style,
        }
        with (pack_dir / "pair_index.json").open("w", encoding="utf-8") as f:
            json.dump(pair_index, f, indent=2)
        
        print(
            f"  {pack_id}: saved dataset "
            f"train={len(train_filenames)} val={len(val_filenames)} "
            f"all_target={len(all_target_filenames)}"
        )
    
    print("Preprocessing complete.")
