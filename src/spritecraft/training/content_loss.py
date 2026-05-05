"""Content preservation losses for texture transfer."""

import torch
import torch.nn.functional as F


def _sobel_kernels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Sobel x and y kernels as conv2d weights."""
    # Kernels for RGB images: [out_channels, in_channels, kH, kW]
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    return sobel_x, sobel_y


def compute_edge_map(image: torch.Tensor) -> torch.Tensor:
    """Compute gradient magnitude edge map for an RGB image [B, 3, H, W]."""
    sobel_x, sobel_y = _sobel_kernels(image.device)
    # Apply to each channel independently, then combine
    edges = []
    for c in range(image.shape[1]):
        channel = image[:, c:c+1, :, :]
        grad_x = F.conv2d(channel, sobel_x, padding=1)
        grad_y = F.conv2d(channel, sobel_y, padding=1)
        mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        edges.append(mag)
    # Average across channels
    return torch.stack(edges, dim=1).mean(dim=1)


def content_preservation_loss(
    logits: torch.Tensor,
    content_tokens: torch.Tensor,
    palette: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Penalize structural deviation from the content reference.

    Args:
        logits: [B, vocab_size-1, H, W] predicted token logits.
        content_tokens: [B, H, W] content reference token indices.
        palette: [vocab_size-1, 3] RGB palette as float tensor [0, 255].
        alpha: weight for the content loss.

    Returns:
        Scalar content preservation loss.
    """
    B, C, H, W = logits.shape
    # Convert logits to probability distribution
    probs = F.softmax(logits, dim=1)  # [B, C, H, W]

    # Expected RGB image from predicted distribution
    # palette: [C, 3] -> reshape for broadcasting
    palette_t = palette.to(device=logits.device, dtype=logits.dtype)  # [C, 3]
    palette_t = palette_t.view(1, C, 3, 1, 1)
    probs_expanded = probs.unsqueeze(2)  # [B, C, 1, H, W]
    pred_rgb = (probs_expanded * palette_t).sum(dim=1)  # [B, 3, H, W]

    # Content reference RGB image
    content_rgb = palette[content_tokens.long()]  # [B, H, W, 3]
    content_rgb = content_rgb.permute(0, 3, 1, 2).to(device=logits.device, dtype=logits.dtype)  # [B, 3, H, W]

    # Compute edge maps
    pred_edges = compute_edge_map(pred_rgb)      # [B, 1, H, W]
    content_edges = compute_edge_map(content_rgb)  # [B, 1, H, W]

    # Normalize edges to [0, 1] per image for stability
    pred_edges = pred_edges / (pred_edges.amax(dim=(2, 3), keepdim=True).clamp_min(1.0))
    content_edges = content_edges / (content_edges.amax(dim=(2, 3), keepdim=True).clamp_min(1.0))

    # MSE between edge maps
    edge_loss = F.mse_loss(pred_edges, content_edges)

    # Also encourage color layout similarity (structural, not exact color match)
    # Use grayscale version and compare local variance
    pred_gray = pred_rgb.mean(dim=1, keepdim=True)
    content_gray = content_rgb.mean(dim=1, keepdim=True)

    # Local variance using avg_pool
    pred_var = pred_gray - F.avg_pool2d(pred_gray, kernel_size=3, stride=1, padding=1)
    content_var = content_gray - F.avg_pool2d(content_gray, kernel_size=3, stride=1, padding=1)

    var_loss = F.mse_loss(pred_var, content_var)

    return alpha * (edge_loss + 0.5 * var_loss)
