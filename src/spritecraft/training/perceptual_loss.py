"""Perceptual and multi-scale structural losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleStructuralLoss(nn.Module):
    """Multi-scale structural similarity loss for pixel-art textures.
    
    Compares Laplacian pyramid levels between predicted and target RGB images
    to enforce structural preservation at multiple scales.
    """
    
    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale structural loss.
        
        Args:
            pred_rgb: [B, 3, H, W] predicted RGB image [0, 255]
            target_rgb: [B, 3, H, W] target RGB image [0, 255]
            
        Returns:
            Scalar loss
        """
        loss = torch.tensor(0.0, device=pred_rgb.device)
        pred = pred_rgb
        target = target_rgb
        
        for scale in range(self.num_scales):
            # Compute gradient magnitude at this scale
            pred_edges = self._gradient_magnitude(pred)
            target_edges = self._gradient_magnitude(target)
            
            # MSE on edges
            edge_loss = F.mse_loss(pred_edges, target_edges)
            loss += edge_loss * (2.0 ** scale)
            
            # Also compare local variance
            pred_var = self._local_variance(pred)
            target_var = self._local_variance(target)
            var_loss = F.mse_loss(pred_var, target_var)
            loss += var_loss * (2.0 ** scale) * 0.5
            
            # Downsample for next scale
            if scale < self.num_scales - 1:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)
        
        return loss / self.num_scales
    
    @staticmethod
    def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude for each channel."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        edges = []
        for c in range(x.shape[1]):
            channel = x[:, c:c+1, :, :]
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)
            mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            edges.append(mag)
        
        return torch.stack(edges, dim=1).mean(dim=1)
    
    @staticmethod
    def _local_variance(x: torch.Tensor) -> torch.Tensor:
        """Compute local variance using average pooling."""
        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mean_sq = F.avg_pool2d(x ** 2, kernel_size=3, stride=1, padding=1)
        return mean_sq - mean ** 2


def perceptual_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    palette: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Compute perceptual/structural loss between predicted and target textures.
    
    Args:
        logits: [B, vocab_size-1, H, W] predicted token logits
        target_tokens: [B, H, W] ground truth token indices
        palette: [vocab_size-1, 3] RGB palette as float tensor [0, 255]
        alpha: weight for the perceptual loss
        
    Returns:
        Scalar perceptual loss
    """
    B, C, H, W = logits.shape
    
    # Convert logits to probability distribution
    probs = F.softmax(logits, dim=1)
    
    # Expected RGB from predicted distribution
    palette_t = palette.to(device=logits.device, dtype=logits.dtype)
    palette_t = palette_t.view(1, C, 3, 1, 1)
    probs_expanded = probs.unsqueeze(2)
    pred_rgb = (probs_expanded * palette_t).sum(dim=1)  # [B, 3, H, W]
    
    # Target RGB
    target_rgb = palette[target_tokens.long()]  # [B, H, W, 3]
    target_rgb = target_rgb.permute(0, 3, 1, 2).to(device=logits.device, dtype=logits.dtype)
    
    # Normalize to [0, 1] for stable loss magnitudes
    pred_rgb = pred_rgb / 255.0
    target_rgb = target_rgb / 255.0
    
    # Multi-scale structural loss
    loss_fn = MultiScaleStructuralLoss(num_scales=3)
    loss = loss_fn(pred_rgb, target_rgb)
    
    return alpha * loss
