"""Forward diffusion and masking utilities."""

import torch
import numpy as np

from spritecraft.config import NUM_TIMESTEPS, MASK_TOKEN


def mask_schedule(t: torch.Tensor, T: int = NUM_TIMESTEPS) -> torch.Tensor:
    """Cosine mask probability schedule."""
    return 1 - torch.cos(0.5 * np.pi * (t / T))


def apply_mask(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply absorbing-state mask to target image."""
    # TODO: Implement masking
    raise NotImplementedError("Masking not yet implemented")
