"""Support-retrieval helpers for vanilla-anchored style transfer."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

TEXTURE_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wood", ("planks", "log", "wood", "stem", "hyphae", "bark")),
    ("foliage", ("leaves", "vine", "grass", "moss", "azalea", "sapling")),
    ("ore", ("ore", "raw_", "debris")),
    ("brick", ("brick", "bricks", "tile", "tiles")),
    ("glass", ("glass", "pane")),
    ("soil", ("dirt", "mud", "sand", "gravel", "clay", "farmland", "path")),
    ("stone", ("stone", "cobble", "slate", "calcite", "tuff", "basalt", "andesite", "granite", "diorite")),
    ("ceramic", ("terracotta", "glazed")),
)

SURFACE_ROLE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("top", ("_top",)),
    ("bottom", ("_bottom",)),
    ("side", ("_side",)),
    ("end", ("_end",)),
)

SEMANTIC_STOP_TOKENS = frozenset({
    "block",
    "top",
    "bottom",
    "side",
    "end",
    "front",
    "back",
    "powered",
    "lit",
    "open",
    "smooth",
    "polished",
    "chiseled",
    "cracked",
    "cut",
    "waxed",
    "weathered",
    "oxidized",
    "exposed",
    "stripped",
    "light",
    "dark",
})

TOKEN_ALIASES = {
    "bricks": "brick",
    "tiles": "tile",
    "planks": "plank",
    "leaves": "leaf",
}


def _normalize_name(filename: str) -> str:
    return Path(filename).stem.lower().replace("-", "_").replace(" ", "_")


def infer_texture_family(filename: str) -> str:
    """Infer a coarse texture family from filename tokens."""
    normalized = _normalize_name(filename)
    for family, tokens in TEXTURE_FAMILY_RULES:
        if any(token in normalized for token in tokens):
            return family
    return "generic"


def infer_surface_role(filename: str) -> str:
    """Infer a coarse surface role from filename suffixes."""
    normalized = _normalize_name(filename)
    for role, suffixes in SURFACE_ROLE_RULES:
        if any(normalized.endswith(suffix) for suffix in suffixes):
            return role
    return "core"


def _semantic_tokens(filename: str) -> set[str]:
    """Extract meaningful lexical tokens from a texture filename."""
    tokens = set()
    for raw_token in _normalize_name(filename).split("_"):
        token = TOKEN_ALIASES.get(raw_token, raw_token)
        if not token or token in SEMANTIC_STOP_TOKENS:
            continue
        tokens.add(token)
    return tokens


def compute_texture_descriptor(image_array: np.ndarray) -> np.ndarray:
    """Build a simple, inspectable descriptor for a texture."""
    rgb = image_array.astype(np.float32) / 255.0
    gray = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-6
    grad_orientation = np.mod(np.arctan2(grad_y, grad_x), math.pi)

    orientation_hist, _ = np.histogram(
        grad_orientation,
        bins=8,
        range=(0.0, math.pi),
        weights=grad_mag,
    )
    orientation_hist = orientation_hist.astype(np.float32)
    orientation_hist /= orientation_hist.sum() + 1e-6

    channel_hist_parts = []
    for channel_index in range(3):
        hist, _ = np.histogram(rgb[:, :, channel_index], bins=4, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        channel_hist_parts.append(hist / (hist.sum() + 1e-6))
    channel_hist = np.concatenate(channel_hist_parts, axis=0)

    freq_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    freq_spectrum /= freq_spectrum.max() + 1e-6
    height, width = gray.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = max(radius.max(), 1.0)

    radial_parts = []
    for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        mask = (radius / max_radius >= low) & (radius / max_radius < high)
        if mask.any():
            radial_parts.append(np.array([freq_spectrum[mask].mean()], dtype=np.float32))
        else:
            radial_parts.append(np.zeros(1, dtype=np.float32))
    radial_features = np.concatenate(radial_parts, axis=0)

    gray_bins = np.clip((gray * 15).astype(np.int32), 0, 15)
    gray_hist = np.bincount(gray_bins.reshape(-1), minlength=16).astype(np.float32)
    gray_prob = gray_hist / (gray_hist.sum() + 1e-6)
    entropy = -np.sum(gray_prob * np.log(gray_prob + 1e-6), dtype=np.float32)

    stats = np.array(
        [
            rgb.mean(),
            rgb.std(),
            gray.std(),
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y)),
            np.mean(grad_mag),
            entropy,
        ],
        dtype=np.float32,
    )

    descriptor = np.concatenate([orientation_hist, channel_hist, radial_features, stats], axis=0)
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    return descriptor.astype(np.float32)


def rank_support_candidates(
    target_filename: str,
    candidate_filenames: list[str],
    descriptors: dict[str, np.ndarray],
) -> list[str]:
    """Rank candidate supports by content similarity plus filename priors."""
    target_descriptor = descriptors.get(target_filename)
    if target_descriptor is None:
        return []

    target_family = infer_texture_family(target_filename)
    target_role = infer_surface_role(target_filename)
    target_tokens = _semantic_tokens(target_filename)
    family_matched_candidates = [
        candidate_filename
        for candidate_filename in candidate_filenames
        if infer_texture_family(candidate_filename) == target_family
    ]
    restricted_candidates = (
        family_matched_candidates
        if target_family != "generic" and len(family_matched_candidates) >= 3
        else candidate_filenames
    )
    scored_candidates: list[tuple[float, str]] = []

    for candidate_filename in restricted_candidates:
        candidate_descriptor = descriptors.get(candidate_filename)
        if candidate_descriptor is None:
            continue

        similarity = float(np.dot(target_descriptor, candidate_descriptor))
        candidate_family = infer_texture_family(candidate_filename)
        candidate_tokens = _semantic_tokens(candidate_filename)
        shared_tokens = target_tokens & candidate_tokens

        if candidate_family == target_family and target_family != "generic":
            similarity += 0.12
        if infer_surface_role(candidate_filename) == target_role and target_role != "core":
            similarity += 0.04
        if shared_tokens:
            similarity += 0.18 * len(shared_tokens)
        if candidate_filename == target_filename:
            continue
        scored_candidates.append((similarity, candidate_filename))

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    return [candidate_filename for _score, candidate_filename in scored_candidates]
