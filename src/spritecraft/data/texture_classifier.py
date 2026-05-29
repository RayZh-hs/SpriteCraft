"""Texture complexity classifier for model selection.

Pre-classifies textures as diffusion-friendly (standard, homogeneous)
vs recolor-required (complex, structured) to avoid wasteful diffusion
attempts on textures where diffusion is known to produce poor results.
"""

from __future__ import annotations

import math
import os

import numpy as np

from spritecraft.data.support_index import infer_texture_family

# Thresholds calibrated on 32x32 pixel-art textures.
# Values derived from analyzing descriptor distributions across families.
ENTROPY_MAX = math.log(16)  # max possible entropy for 16-bin histogram
HIGH_FREQ_ENERGY_THRESHOLD = 0.12  # radial bands 3+4 combined, above this = fine detail
GRADIENT_MAG_THRESHOLD = 0.15      # mean gradient magnitude cutoff
ENTROPY_THRESHOLD = 1.0            # entropy above this suggests high complexity
STD_THRESHOLD = 0.18               # grayscale std above this = busy pattern

# Textures in these families typically benefit from diffusion
DIFFUSION_FAVORED_FAMILIES = frozenset({
    "ore", "wood", "stone", "soil", "brick", "ceramic", "glass",
})

# Known complex textures that should always use recolor
RECOLOR_ONLY_PATTERNS = frozenset({
    "bookshelf", "tnt_side", "tnt_top", "tnt_bottom",
})

# Texture families normally unfavored for diffusion
RECOLOR_FAVORED_FAMILIES = frozenset({
    "foliage",
})


def _box_mean(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute box-filtered mean using simple sliding window (no scipy dependency)."""
    h, w = gray.shape
    pad = ksize // 2
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="edge")
    # Use strided views for efficiency
    shape = (h, w, ksize, ksize)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return windows.mean(axis=(2, 3))


def classify_texture_complexity(
    image_array: np.ndarray,
    filename: str = "",
) -> dict[str, float | str | bool]:
    """Analyze a vanilla content texture and return a classification verdict.

    Args:
        image_array: uint8 RGB image array [H, W, 3]
        filename: texture filename for family heuristics

    Returns:
        dict with keys:
            - diffusion_score: float [0, 1], higher = better for diffusion
            - use_diffusion: bool, recommended model choice
            - use_recolor: bool, alternative recommended
            - entropy: float, luminance entropy
            - high_freq_energy: float, FFT high-band energy
            - grad_mag_mean: float, mean gradient magnitude
            - grad_std: float, std dev of gradient magnitude
            - local_std_mean: float, mean local std
            - texture_family: str, inferred family
            - complexity_label: str, "simple" | "moderate" | "complex"
    """
    # Compute full descriptor for feature extraction
    rgb = image_array.astype(np.float32) / 255.0
    gray = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-6

    # 1. Entropy (from 16-bin luminance histogram)
    gray_bins = np.clip((gray * 15).astype(np.int32), 0, 15)
    gray_hist = np.bincount(gray_bins.reshape(-1), minlength=16).astype(np.float32)
    gray_prob = gray_hist / (gray_hist.sum() + 1e-6)
    entropy = float(-np.sum(gray_prob * np.log(gray_prob + 1e-6)))

    # 2. FFT radial energy (high-frequency = complex detail)
    freq_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    freq_spectrum /= freq_spectrum.max() + 1e-6
    height, width = gray.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = max(radius.max(), 1.0)

    high_freq_mask = (radius / max_radius >= 0.5) & (radius / max_radius <= 1.01)
    high_freq_energy = float(freq_spectrum[high_freq_mask].mean()) if high_freq_mask.any() else 0.0
    mid_freq_mask = (radius / max_radius >= 0.25) & (radius / max_radius < 0.5)
    mid_freq_energy = float(freq_spectrum[mid_freq_mask].mean()) if mid_freq_mask.any() else 0.0

    # 3. Gradient statistics
    grad_mag_mean = float(np.mean(grad_mag))
    grad_mag_std = float(np.std(grad_mag))

    # 4. Local standard deviation (windowed contrast)
    local_mean = _box_mean(gray, ksize=3)
    local_sq_mean = _box_mean(gray ** 2, ksize=3)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var + 1e-6)
    local_std_mean = float(local_std.mean())
    local_std_max = float(local_std.max())

    # 5. Texture family
    texture_family = infer_texture_family(filename) if filename else "generic"

    # === COMPUTE DIFFUSION SUITABILITY SCORE ===

    # Entropy penalty: high entropy = complex = bad for diffusion
    entropy_penalty = 0.0
    if entropy > ENTROPY_THRESHOLD:
        excess = min(entropy - ENTROPY_THRESHOLD, ENTROPY_MAX - ENTROPY_THRESHOLD)
        entropy_penalty = 0.30 * (excess / (ENTROPY_MAX - ENTROPY_THRESHOLD)) ** 0.7

    # High-frequency penalty: lots of fine detail = hard for diffusion
    hf_penalty = 0.0
    if high_freq_energy > HIGH_FREQ_ENERGY_THRESHOLD:
        excess = high_freq_energy - HIGH_FREQ_ENERGY_THRESHOLD
        hf_penalty = 0.25 * min(excess / 0.15, 1.0)

    # Mid-frequency penalty (structure/texture patterns like tiled ores, planks)
    mf_penalty = mid_freq_energy * 0.10

    # Gradient penalty: many edges = complex structure
    grad_penalty = 0.0
    if grad_mag_mean > GRADIENT_MAG_THRESHOLD:
        excess = grad_mag_mean - GRADIENT_MAG_THRESHOLD
        grad_penalty = 0.20 * min(excess / 0.10, 1.0)

    # Gradient variability penalty: uneven detail distribution
    grad_var_penalty = min(grad_mag_std * 1.2, 0.15)

    # Local std penalty: busy regions
    std_penalty = 0.0
    if local_std_mean > STD_THRESHOLD:
        excess = local_std_mean - STD_THRESHOLD
        std_penalty = 0.10 * min(excess / 0.12, 1.0)

    # Base score from content features
    base_features_score = (1.0
        - entropy_penalty
        - hf_penalty
        - mf_penalty
        - grad_penalty
        - grad_var_penalty
        - std_penalty
    )

    # === FAMILY-BASED ADJUSTMENTS ===

    family_name = os.path.splitext(os.path.basename(filename))[0].lower().replace("-", "_").replace(" ", "_") if filename else ""
    family_bonus = 0.0

    # Hard override: known complex patterns always use recolor
    if any(pattern in family_name for pattern in RECOLOR_ONLY_PATTERNS):
        family_bonus = -0.80

    elif texture_family in RECOLOR_FAVORED_FAMILIES:
        family_bonus = -0.25
    elif texture_family in DIFFUSION_FAVORED_FAMILIES:
        family_bonus = 0.10
    elif texture_family == "generic":
        if entropy > ENTROPY_THRESHOLD * 0.8:
            family_bonus = -0.15

    # Additional filename heuristics
    if "stained" in family_name or "glass" in family_name:
        family_bonus -= 0.10
    if "daylight" in family_name or "detector" in family_name:
        family_bonus -= 0.10

    diffusion_score = max(0.0, min(1.0, base_features_score + family_bonus))

    # === CLASSIFICATION ===

    DIFFUSION_THRESHOLD = 0.45

    use_diffusion = diffusion_score >= DIFFUSION_THRESHOLD
    use_recolor = not use_diffusion

    if use_diffusion:
        complexity_label = "simple" if diffusion_score > 0.75 else "moderate"
    else:
        complexity_label = "complex"

    return {
        "diffusion_score": round(diffusion_score, 4),
        "use_diffusion": use_diffusion,
        "use_recolor": use_recolor,
        "entropy": round(entropy, 4),
        "high_freq_energy": round(high_freq_energy, 4),
        "mid_freq_energy": round(mid_freq_energy, 4),
        "grad_mag_mean": round(grad_mag_mean, 4),
        "grad_std": round(grad_mag_std, 4),
        "local_std_mean": round(local_std_mean, 4),
        "local_std_max": round(local_std_max, 4),
        "texture_family": texture_family,
        "complexity_label": complexity_label,
    }


def batch_classify(
    image_arrays: np.ndarray,
    filenames: list[str],
) -> list[dict[str, float | str | bool]]:
    """Classify a batch of textures.

    Args:
        image_arrays: [N, H, W, 3] uint8 RGB array
        filenames: list of N filenames

    Returns:
        list of classification dicts
    """
    results = []
    for i, filename in enumerate(filenames):
        img = image_arrays[i] if image_arrays.ndim == 4 else image_arrays
        result = classify_texture_complexity(img, filename)
        results.append(result)
    return results

