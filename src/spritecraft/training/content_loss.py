"""Content-aware RGB losses for structure-preserving texture transfer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

STRUCTURE_LOSS_WEIGHT = 1.0
CONTRAST_LOSS_WEIGHT = 1.75
CONTRAST_GATE_GAMMA = 2.0
DETAIL_EMPHASIS_SCALE = 1.5
DETAIL_TOP_FRACTION = 0.125
LAPLACIAN_STRUCTURE_WEIGHT = 0.75
EDGE_SHORTFALL_WEIGHT = 0.35


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Project RGB images in [0, 1] to a single luminance channel."""
    return (
        0.299 * rgb[:, 0:1]
        + 0.587 * rgb[:, 1:2]
        + 0.114 * rgb[:, 2:3]
    )


def _directional_gradients(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute simple forward differences on a single-channel image."""
    grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    return grad_x, grad_y


def _gradient_magnitude_map(grad_x: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
    """Convert directional forward differences into an HxW magnitude map."""
    grad_x_full = F.pad(grad_x, (0, 1, 0, 0))
    grad_y_full = F.pad(grad_y, (0, 0, 0, 1))
    return torch.sqrt(grad_x_full.square() + grad_y_full.square() + 1e-6)


def _high_pass(gray: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Return a compact high-pass residual used as a detail proxy."""
    blurred = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return gray - blurred


def _laplacian_response(gray: torch.Tensor) -> torch.Tensor:
    """Return a 4-neighbor Laplacian response map for edge/detail supervision."""
    kernel = gray.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
    ).view(1, 1, 3, 3)
    return F.conv2d(gray, kernel, padding=1)


def _local_std(gray: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Estimate local contrast via windowed luminance standard deviation."""
    mean = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    mean_sq = F.avg_pool2d(gray.square(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    variance = (mean_sq - mean.square()).clamp_min(0.0)
    return torch.sqrt(variance + 1e-6)


def _detail_emphasis(target_std: torch.Tensor) -> torch.Tensor:
    """Raise the penalty for textures whose target contains concentrated detail."""
    flat = target_std.flatten(start_dim=1)
    top_count = max(1, int(flat.shape[1] * DETAIL_TOP_FRACTION))
    top_values = torch.topk(flat, k=top_count, dim=1).values
    max_value = flat.amax(dim=1, keepdim=True).clamp_min(1e-4)
    detail_strength = (top_values.mean(dim=1, keepdim=True) / max_value).clamp(0.0, 1.0)
    emphasis = 1.0 + DETAIL_EMPHASIS_SCALE * detail_strength
    return emphasis.view(-1, 1, 1, 1)


def _build_content_state(
    pred_rgb: torch.Tensor,
    content_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute reusable scalar terms and map tensors for content-aware losses."""
    pred_gray = _luminance(pred_rgb)
    content_gray = _luminance(content_rgb)
    target_gray = _luminance(target_rgb)

    pred_grad_x, pred_grad_y = _directional_gradients(pred_gray)
    content_grad_x, content_grad_y = _directional_gradients(content_gray)
    target_grad_x, target_grad_y = _directional_gradients(target_gray)

    gradient_delta_loss = (
        F.l1_loss(pred_grad_x - content_grad_x, target_grad_x - content_grad_x)
        + F.l1_loss(pred_grad_y - content_grad_y, target_grad_y - content_grad_y)
    )
    pred_grad_delta_mag = _gradient_magnitude_map(
        pred_grad_x - content_grad_x,
        pred_grad_y - content_grad_y,
    )
    target_grad_delta_mag = _gradient_magnitude_map(
        target_grad_x - content_grad_x,
        target_grad_y - content_grad_y,
    )

    pred_detail = _high_pass(pred_gray)
    content_detail = _high_pass(content_gray)
    target_detail = _high_pass(target_gray)
    pred_detail_delta = pred_detail - content_detail
    target_detail_delta = target_detail - content_detail
    detail_delta_loss = F.l1_loss(
        pred_detail_delta,
        target_detail_delta,
    )
    pred_laplacian = _laplacian_response(pred_gray)
    content_laplacian = _laplacian_response(content_gray)
    target_laplacian = _laplacian_response(target_gray)
    pred_laplacian_delta = pred_laplacian - content_laplacian
    target_laplacian_delta = target_laplacian - content_laplacian
    laplacian_delta_loss = F.l1_loss(pred_laplacian_delta, target_laplacian_delta)
    structure_loss = (
        gradient_delta_loss
        + 0.5 * detail_delta_loss
        + LAPLACIAN_STRUCTURE_WEIGHT * laplacian_delta_loss
    )

    pred_std = 0.5 * (_local_std(pred_gray, kernel_size=3) + _local_std(pred_gray, kernel_size=5))
    target_std = 0.5 * (_local_std(target_gray, kernel_size=3) + _local_std(target_gray, kernel_size=5))

    # Gate each spatial location by the target's local contrast energy so flat
    # target regions are effectively ignored. A squared gate focuses the penalty
    # on the highest-detail target regions instead of spreading it evenly.
    max_target_std = target_std.amax(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    gate = (target_std / max_target_std).clamp(0.0, 1.0).pow(CONTRAST_GATE_GAMMA)
    detail_emphasis = _detail_emphasis(target_std)

    # Only penalize under-contrast. Over-sharpening is already constrained by
    # direct reconstruction and structure losses.
    under_contrast = F.relu(target_std - pred_std)
    weighted = detail_emphasis * gate * under_contrast
    normalizer = (detail_emphasis * gate).sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    contrast_loss = (weighted.sum(dim=(1, 2, 3), keepdim=True) / normalizer).mean()

    # Penalize edges that stay weaker than the target even when local standard
    # deviation is similar. This catches soft mortar lines and blurred texel
    # boundaries without rewarding arbitrary oversharpening.
    edge_shortfall = F.relu(target_grad_delta_mag - pred_grad_delta_mag)
    weighted_edge_shortfall = detail_emphasis * gate * edge_shortfall
    edge_shortfall_loss = (
        weighted_edge_shortfall.sum(dim=(1, 2, 3), keepdim=True) / normalizer
    ).mean()
    contrast_loss = contrast_loss + EDGE_SHORTFALL_WEIGHT * edge_shortfall_loss

    return {
        "pred_gray": pred_gray,
        "content_gray": content_gray,
        "target_gray": target_gray,
        "pred_local_std": pred_std,
        "target_local_std": target_std,
        "contrast_gate": gate,
        "detail_emphasis": detail_emphasis,
        "under_contrast": under_contrast,
        "weighted_under_contrast": weighted,
        "pred_detail_delta": pred_detail_delta,
        "target_detail_delta": target_detail_delta,
        "pred_laplacian_delta": pred_laplacian_delta,
        "target_laplacian_delta": target_laplacian_delta,
        "pred_grad_delta_mag": pred_grad_delta_mag,
        "target_grad_delta_mag": target_grad_delta_mag,
        "gradient_delta_loss": gradient_delta_loss,
        "detail_delta_loss": detail_delta_loss,
        "laplacian_delta_loss": laplacian_delta_loss,
        "structure_loss": structure_loss,
        "contrast_loss": contrast_loss,
        "edge_shortfall": edge_shortfall,
        "weighted_edge_shortfall": weighted_edge_shortfall,
        "edge_shortfall_loss": edge_shortfall_loss,
    }


def source_relative_structure_loss(
    pred_rgb: torch.Tensor,
    content_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> torch.Tensor:
    """Match the structural change from source to target rather than target alone."""
    return _build_content_state(pred_rgb, content_rgb, target_rgb)["structure_loss"]


def target_conditioned_contrast_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> torch.Tensor:
    """Penalize missing local contrast only where the target actually contains it."""
    return _build_content_state(pred_rgb, pred_rgb, target_rgb)["contrast_loss"]


def rgb_content_loss_components(
    pred_rgb: torch.Tensor,
    content_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return scalar content-loss components for logging and weighting."""
    state = _build_content_state(pred_rgb, content_rgb, target_rgb)
    total = (
        STRUCTURE_LOSS_WEIGHT * state["structure_loss"] +
        CONTRAST_LOSS_WEIGHT * state["contrast_loss"]
    )
    return {
        "content_structure_loss": state["structure_loss"],
        "content_gradient_delta_loss": state["gradient_delta_loss"],
        "content_detail_delta_loss": state["detail_delta_loss"],
        "content_contrast_loss": state["contrast_loss"],
        "content_loss": total,
    }


def rgb_content_diagnostic_maps(
    pred_rgb: torch.Tensor,
    content_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return spatial maps that explain why the content loss is firing."""
    state = _build_content_state(pred_rgb, content_rgb, target_rgb)
    return {
        "pred_local_std": state["pred_local_std"],
        "target_local_std": state["target_local_std"],
        "contrast_gate": state["contrast_gate"],
        "detail_emphasis": state["detail_emphasis"],
        "under_contrast": state["under_contrast"],
        "weighted_under_contrast": state["weighted_under_contrast"],
        "edge_shortfall": state["edge_shortfall"],
        "weighted_edge_shortfall": state["weighted_edge_shortfall"],
        "pred_detail_delta": state["pred_detail_delta"].abs(),
        "target_detail_delta": state["target_detail_delta"].abs(),
        "detail_delta_gap": (state["pred_detail_delta"] - state["target_detail_delta"]).abs(),
        "pred_laplacian_delta": state["pred_laplacian_delta"].abs(),
        "target_laplacian_delta": state["target_laplacian_delta"].abs(),
        "laplacian_delta_gap": (state["pred_laplacian_delta"] - state["target_laplacian_delta"]).abs(),
        "pred_grad_delta_mag": state["pred_grad_delta_mag"],
        "target_grad_delta_mag": state["target_grad_delta_mag"],
        "grad_delta_gap": (state["pred_grad_delta_mag"] - state["target_grad_delta_mag"]).abs(),
    }


def rgb_content_loss(
    pred_rgb: torch.Tensor,
    content_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> torch.Tensor:
    """Combined content loss for preserving source structure and target detail."""
    components = rgb_content_loss_components(pred_rgb, content_rgb, target_rgb)
    return components["content_loss"]
