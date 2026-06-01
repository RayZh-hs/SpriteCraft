"""Post-hoc diffusion output quality evaluator for model selection.

Instead of pre-classifying textures, this module evaluates the actual
diffusion output against the vanilla content image to decide whether
the result is satisfactory or whether we should fall back to RecolorNet.

Bad diffusion outputs exhibit: structural divergence, edge loss, detail
mismatch, excessive noise. These traits are detected and scored.
"""

from __future__ import annotations

import numpy as np


HIGH_PATTERN_EDGE = 0.09
HIGH_PATTERN_DETAIL = 0.05
HIGH_PATTERN_COMPLEXITY = 0.50


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert [H, W, 3] float RGB to luminance."""
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _gradient_magnitude(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward differences, returns grad_x[H,W-1], grad_y[H-1,W], grad_mag[H,W]."""
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    mag = np.sqrt(
        np.pad(gx, ((0, 0), (0, 1))) ** 2 + np.pad(gy, ((0, 1), (0, 0))) ** 2
    ) + 1e-8
    return gx, gy, mag


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """L2-normalized dot product."""
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(an, bn))


def _high_pass_residual(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Detail proxy: gray minus 3x3 box blur."""
    h, w = gray.shape
    pad = ksize // 2
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="edge")
    shape = (h, w, ksize, ksize)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    blurred = windows.mean(axis=(2, 3))
    return gray - blurred


def _ratio_window_score(
    ratio: float,
    *,
    low: float,
    high: float,
    min_ratio: float,
    max_ratio: float,
) -> float:
    """Score a ratio against a preferred window with linear falloff."""
    if min_ratio >= low or max_ratio <= high:
        raise ValueError("Expected min_ratio < low < high < max_ratio")

    if ratio < min_ratio or ratio > max_ratio:
        return 0.0
    if ratio < low:
        return (ratio - min_ratio) / (low - min_ratio)
    if ratio > high:
        return (max_ratio - ratio) / (max_ratio - high)
    return 1.0


def evaluate_diffusion_output(
    diffusion_rgb: np.ndarray,
    content_rgb: np.ndarray,
) -> dict[str, float]:
    """Score whether a diffusion output is structurally sound vs the content.

    Args:
        diffusion_rgb: [H, W, 3] uint8 diffusion-generated image
        content_rgb:   [H, W, 3] uint8 vanilla content image

    Returns:
        Dict with quality score (0-1, higher=better) and diagnostic traits.
        key "use_diffusion": bool, True if diffusion output is acceptable.
    """
    d = diffusion_rgb.astype(np.float32) / 255.0
    c = content_rgb.astype(np.float32) / 255.0

    d_gray = _to_gray(d)
    c_gray = _to_gray(c)

    # === 1. Gradient alignment ===
    # How well does the output preserve the content's edge structure?
    _, _, d_mag = _gradient_magnitude(d_gray)
    _, _, c_mag = _gradient_magnitude(c_gray)
    grad_alignment = _cosine_similarity(d_mag.flatten(), c_mag.flatten())

    # === 2. Edge energy ratio ===
    # Significant edge loss = smeared/blurred output (bad sign for diffusion).
    # Significant edge gain = random noise (also bad).
    d_edge = d_mag.mean()
    c_edge = c_mag.mean()
    if c_edge > 1e-6:
        edge_ratio = d_edge / c_edge
        # Penalize both edge loss (< 0.5x) and edge explosion (> 2.0x)
        if edge_ratio < 0.5:
            edge_fidelity = edge_ratio / 0.5  # 0→0, 0.5→1
        elif edge_ratio > 2.0:
            edge_fidelity = max(0.0, (4.0 - edge_ratio) / 2.0)  # 2→1, 4→0
        else:
            edge_fidelity = 1.0
    else:
        edge_fidelity = 1.0

    # === 3. Detail coherence ===
    # Does the high-pass residual pattern match? Complex textures (bookshelf,
    # TNT) produce randomized diffusion details that don't match content.
    d_detail = _high_pass_residual(d_gray)
    c_detail = _high_pass_residual(c_gray)
    detail_coherence = _cosine_similarity(d_detail.flatten(), c_detail.flatten())

    # === 3.5. Detail energy fidelity (symmetric: detects both blur and noise) ===
    d_detail_energy = float(np.abs(d_detail).mean())
    c_detail_energy = float(np.abs(c_detail).mean())
    if c_detail_energy > 1e-8:
        detail_energy_ratio = d_detail_energy / (c_detail_energy + 1e-8)
        if detail_energy_ratio < 0.3:
            detail_fidelity = detail_energy_ratio / 0.3  # blur penalty: 0→0, 0.3→1
        elif detail_energy_ratio > 1.3:
            detail_fidelity = max(0.0, 1.0 - (detail_energy_ratio - 1.3))  # noise penalty
        else:
            detail_fidelity = 1.0
    else:
        detail_energy_ratio = 1.0
        detail_fidelity = 1.0

    # === 4. Entropy delta ===
    # Large entropy increase = noise injection. Large drop = washed out.
    c_hist = np.histogram(c_gray, bins=16, range=(0, 1))[0].astype(np.float32)
    d_hist = np.histogram(d_gray, bins=16, range=(0, 1))[0].astype(np.float32)
    c_prob = c_hist / (c_hist.sum() + 1e-8)
    d_prob = d_hist / (d_hist.sum() + 1e-8)
    c_entropy = float(-np.sum(c_prob * np.log(c_prob + 1e-8)))
    d_entropy = float(-np.sum(d_prob * np.log(d_prob + 1e-8)))
    ent_delta = d_entropy - c_entropy
    # Penalize entropy gain > 0.5 (noise) or loss > 1.0 (washed out)
    if ent_delta > 0.5:
        ent_score = max(0.0, 1.0 - (ent_delta - 0.5))
    elif ent_delta < -1.0:
        ent_score = max(0.0, 1.0 + (ent_delta + 1.0) * 0.5)
    else:
        ent_score = 1.0

    # === 5. Structural simplicity (from content) ===
    # Measure how structured the content is. If content is very simple
    # (low entropy, low edge energy, smooth), then strong gradient
    # alignment is expected and the bar is higher.
    c_entropy_norm = c_entropy / np.log(16)  # [0,1]
    c_edge_norm = min(c_edge / 0.25, 1.0)    # normalize
    content_complexity = 0.5 * (c_entropy_norm + c_edge_norm)
    high_pattern_content = (
        content_complexity >= HIGH_PATTERN_COMPLEXITY
        and c_edge >= HIGH_PATTERN_EDGE
        and c_detail_energy >= HIGH_PATTERN_DETAIL
    )

    high_pattern_gate = 1.0
    pattern_edge_fidelity = 1.0
    pattern_detail_fidelity = 1.0
    if high_pattern_content:
        # Textures with strong line art / repeated bands (for example TNT sides)
        # should not pass on "roughly enough edges" alone. They need aligned
        # detail and a tighter edge/detail energy match.
        pattern_edge_fidelity = _ratio_window_score(
            edge_ratio,
            low=0.78,
            high=1.18,
            min_ratio=0.55,
            max_ratio=1.45,
        )
        pattern_detail_fidelity = _ratio_window_score(
            detail_energy_ratio,
            low=0.80,
            high=1.16,
            min_ratio=0.55,
            max_ratio=1.35,
        )
        positive_detail_gate = max(0.0, min(1.0, detail_coherence / 0.35))
        high_pattern_gate = min(
            pattern_edge_fidelity,
            pattern_detail_fidelity,
            positive_detail_gate,
        )

    # === Composite quality score ===
    # Weight components to emphasize structural preservation.
    # Gradient alignment is most important — it directly measures
    # whether the output "respects" the content's structure.
    quality = (
        0.30 * grad_alignment
        + 0.20 * edge_fidelity
        + 0.18 * detail_coherence
        + 0.17 * detail_fidelity
        + 0.10 * ent_score
        + 0.03  # modest baseline to keep scores anchored
    )
    if high_pattern_content:
        quality *= 0.55 + 0.45 * high_pattern_gate

    # Adjust threshold by content complexity:
    # - Simple content (ores, stone): need high quality to accept diffusion
    # - Complex content (leaves, bookshelf): more lenient but still strict
    if content_complexity < 0.3:
        threshold = 0.60
    elif content_complexity < 0.5:
        threshold = 0.55
    else:
        threshold = 0.50
    if high_pattern_content:
        threshold = max(threshold, 0.68)

    use_diffusion = quality >= threshold and (not high_pattern_content or detail_coherence > 0.0)

    return {
        "quality_score": round(quality, 4),
        "use_diffusion": use_diffusion,
        "use_recolor": not use_diffusion,
        "grad_alignment": round(grad_alignment, 4),
        "edge_fidelity": round(edge_fidelity, 4),
        "detail_coherence": round(detail_coherence, 4),
        "detail_fidelity": round(detail_fidelity, 4),
        "detail_energy_ratio": round(detail_energy_ratio, 4),
        "entropy_score": round(ent_score, 4),
        "content_complexity": round(content_complexity, 4),
        "high_pattern_content": high_pattern_content,
        "high_pattern_gate": round(high_pattern_gate, 4),
        "pattern_edge_fidelity": round(pattern_edge_fidelity, 4),
        "pattern_detail_fidelity": round(pattern_detail_fidelity, 4),
        "content_entropy": round(c_entropy, 4),
        "diff_entropy": round(d_entropy, 4),
        "content_edge": round(c_edge, 6),
        "diff_edge": round(d_edge, 6),
        "entropy_delta": round(ent_delta, 4),
        "threshold": threshold,
    }
