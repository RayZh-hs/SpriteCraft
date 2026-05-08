"""Forward diffusion and denoising utilities for continuous RGB."""

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


def predict_x0_from_noise(
    noisy_x: torch.Tensor,
    pred_noise: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Recover x0 from xt and predicted noise."""
    B = noisy_x.shape[0]
    device = noisy_x.device
    t_idx = t - 1
    alpha_cumprod_t = alphas_cumprod[t_idx].view(B, 1, 1, 1).to(device)
    return (noisy_x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)


def ddpm_sample_step(
    noisy_x: torch.Tensor,
    pred_noise: torch.Tensor,
    t: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    clip_x0: bool = True,
) -> torch.Tensor:
    """Single DDPM sampling step for epsilon-prediction models."""
    B = noisy_x.shape[0]
    device = noisy_x.device
    t_idx = t - 1

    beta_t = betas[t_idx].view(B, 1, 1, 1).to(device)
    alpha_t = alphas[t_idx].view(B, 1, 1, 1).to(device)
    alpha_cumprod_t = alphas_cumprod[t_idx].view(B, 1, 1, 1).to(device)

    alpha_cumprod_prev = torch.ones_like(alpha_cumprod_t)
    has_previous = t > 1
    if has_previous.any():
        prev_t_idx = (t[has_previous] - 2).long()
        alpha_cumprod_prev[has_previous] = alphas_cumprod[prev_t_idx].view(-1, 1, 1, 1).to(device)

    pred_x0 = predict_x0_from_noise(noisy_x, pred_noise, t, alphas_cumprod)
    if clip_x0:
        pred_x0 = torch.clamp(pred_x0, 0.0, 1.0)

    coef_x0 = (torch.sqrt(alpha_cumprod_prev) * beta_t) / (1 - alpha_cumprod_t)
    coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev)) / (1 - alpha_cumprod_t)
    model_mean = coef_x0 * pred_x0 + coef_xt * noisy_x

    posterior_variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
    noise = torch.randn_like(noisy_x)
    nonzero_mask = (t > 1).view(B, 1, 1, 1)
    sampled = model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_variance, min=1e-12)) * noise
    return sampled
