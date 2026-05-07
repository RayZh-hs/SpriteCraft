"""Forward diffusion and denoising utilities for continuous RGB."""

import math

import torch

from spritecraft.config import NUM_TIMESTEPS


def get_beta_schedule(T: int = NUM_TIMESTEPS, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule for Gaussian diffusion."""
    return torch.linspace(beta_start, beta_end, T)


def get_alpha_schedule(betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute alpha and alpha_cumprod from betas."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod


def add_noise(x: torch.Tensor, t: torch.Tensor, alphas_cumprod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise to images at timestep t.
    
    Args:
        x: [B, C, H, W] clean images
        t: [B] timesteps (1-indexed, in range [1, T])
        alphas_cumprod: [T] cumulative product of alphas
    
    Returns:
        noisy_x: [B, C, H, W] noisy images
        noise: [B, C, H, W] noise that was added
    """
    B = x.shape[0]
    device = x.device
    
    # t is 1-indexed, convert to 0-indexed for indexing
    t_idx = t - 1
    
    # Get alpha_cumprod for each sample
    alpha_cumprod_t = alphas_cumprod[t_idx].view(B, 1, 1, 1).to(device)
    
    # Sample noise
    noise = torch.randn_like(x)
    
    # Add noise according to diffusion schedule
    noisy_x = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * noise
    
    return noisy_x, noise


def ddim_sample_step(
    model: torch.nn.Module,
    noisy_x: torch.Tensor,
    content_source: torch.Tensor,
    style_refs: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    alphas: torch.Tensor,
    eta: float = 0.0,
) -> torch.Tensor:
    """Single DDIM sampling step.
    
    Args:
        model: denoising model
        noisy_x: [B, C, H, W] current noisy images
        content_source: [B, C, H, W] content reference
        style_refs: [B, N, C, H, W] style references
        t: [B] current timesteps
        alphas_cumprod: [T] cumulative alphas
        alphas: [T] alphas
        eta: DDIM eta parameter (0 = deterministic)
    
    Returns:
        [B, C, H, W] denoised images at t-1
    """
    B = noisy_x.shape[0]
    device = noisy_x.device
    t_idx = t - 1
    
    # Predict noise
    with torch.no_grad():
        pred = model(noisy_x, content_source, style_refs, t)
    
    # For simplicity, we predict the clean image directly (not epsilon)
    # The model outputs denoised image directly
    pred_x0 = pred
    
    alpha_t = alphas[t_idx].view(B, 1, 1, 1).to(device)
    alpha_cumprod_t = alphas_cumprod[t_idx].view(B, 1, 1, 1).to(device)
    
    # Compute previous timestep alpha_cumprod
    prev_t = torch.clamp(t - 1, min=1)
    prev_t_idx = prev_t - 1
    alpha_cumprod_prev = alphas_cumprod[prev_t_idx].view(B, 1, 1, 1).to(device)
    
    # DDIM formula
    pred_noise = (noisy_x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)
    
    # Direction pointing to x_t
    dir_xt = torch.sqrt(1 - alpha_cumprod_prev - eta ** 2 * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)) * pred_noise
    
    # Random noise for stochasticity
    if eta > 0:
        noise = torch.randn_like(noisy_x)
        sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_prev)
        dir_xt = dir_xt + sigma_t * noise
    
    # Compute x_{t-1}
    prev_x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt
    
    return prev_x
