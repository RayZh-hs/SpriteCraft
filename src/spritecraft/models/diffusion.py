"""Forward diffusion and masking utilities."""

import math

import torch

from spritecraft.config import NUM_TIMESTEPS, MASK_TOKEN


def mask_schedule(t: torch.Tensor, T: int = NUM_TIMESTEPS) -> torch.Tensor:
    """Cosine mask probability schedule."""
    t = t.to(dtype=torch.float32)
    return 1.0 - torch.cos(0.5 * math.pi * (t / T))


def apply_mask(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply absorbing-state masking to a batch of target token maps."""
    x = x.long()
    t = t.to(device=x.device).reshape(-1)

    mask_prob = mask_schedule(t).to(device=x.device).clamp_(0.0, 1.0).view(-1, 1, 1)
    mask = torch.rand(x.shape, device=x.device) < mask_prob
    return torch.where(mask, torch.full_like(x, MASK_TOKEN), x)
