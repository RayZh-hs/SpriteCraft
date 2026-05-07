"""Iterative denoising sampler for continuous RGB diffusion."""

from contextlib import nullcontext
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image

from spritecraft.config import (
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    NUM_TIMESTEPS,
    OUTPUT_DIR,
    pack_checkpoint_dir,
)
from spritecraft.models.diffusion import get_beta_schedule, get_alpha_schedule
from spritecraft.models.unet import StyleAwareUNet


class PredictionBundleResult(TypedDict):
    bundle_dir: Path
    original_path: Path
    support_path: Path
    produced_path: Path
    truth_path: Path | None
    comparison_path: Path
    metrics: dict[str, float | int | bool] | None


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    latest_path = checkpoint_dir / "latest.pt"
    if latest_path.exists():
        return latest_path

    step_checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if step_checkpoints:
        return step_checkpoints[-1]

    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def load_model(checkpoint_dir: str | Path, pack_id: str | None = None) -> tuple[StyleAwareUNet, Path]:
    """Load the latest checkpoint for a pack."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if pack_id is not None:
        checkpoint_dir = pack_checkpoint_dir(checkpoint_dir, pack_id)
    
    checkpoint_path = checkpoint_dir if checkpoint_dir.is_file() else _latest_checkpoint_path(checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = _select_device()
    model = StyleAwareUNet(
        in_channels=3,
        style_channels=64,
        base_channels=128,
        num_style_refs=3,
    ).to(device)
    
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with the current model. "
            "Train a fresh model after re-running preprocessing."
        ) from exc
    model.eval()
    return model, checkpoint_path


@torch.no_grad()
def sample_rgb(
    model: StyleAwareUNet,
    content_rgb: torch.Tensor,
    style_refs: torch.Tensor,
    num_steps: int = NUM_TIMESTEPS,
) -> torch.Tensor:
    """Generate RGB texture using iterative denoising.
    
    Args:
        model: trained StyleAwareUNet
        content_rgb: [1, 3, H, W] or [3, H, W] vanilla content
        style_refs: [N, 3, H, W] or [1, N, 3, H, W] style reference textures
        num_steps: number of denoising steps
    
    Returns:
        [3, H, W] generated RGB texture
    """
    device = next(model.parameters()).device
    
    if content_rgb.dim() == 3:
        content_rgb = content_rgb.unsqueeze(0)
    if style_refs.dim() == 4:
        style_refs = style_refs.unsqueeze(0)
    
    content_rgb = content_rgb.to(device)
    style_refs = style_refs.to(device)
    
    # Precompute diffusion schedule
    betas = get_beta_schedule(num_steps).to(device)
    alphas, alphas_cumprod = get_alpha_schedule(betas)
    alphas_cumprod = alphas_cumprod.to(device)
    
    # Start from noise
    x = torch.randn_like(content_rgb)
    
    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    
    # Iterative denoising
    for timestep in range(num_steps, 0, -1):
        t = torch.full((content_rgb.shape[0],), timestep, dtype=torch.long, device=device)
        
        with autocast_context():
            pred = model(x, content_rgb, style_refs, t)
        
        if timestep > 1:
            # DDPM sampling
            alpha_t = alphas_cumprod[timestep - 1]
            alpha_prev = alphas_cumprod[timestep - 2]
            beta_t = 1 - alpha_t / alpha_prev
            
            # Add noise for stochasticity
            noise = torch.randn_like(x)
            x = (x - beta_t / torch.sqrt(1 - alpha_t) * (x - pred)) / torch.sqrt(1 - beta_t)
            x = x + torch.sqrt(beta_t) * noise
        else:
            x = pred
    
    return torch.clamp(x.squeeze(0), 0, 1).cpu()


def compute_metrics(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float | int | bool]:
    """Compute simple texture metrics against a ground-truth texture."""
    pred_np = prediction.cpu().float().numpy()
    truth_np = truth.cpu().float().numpy()
    
    mae = np.abs(pred_np - truth_np).mean()
    mse = ((pred_np - truth_np) ** 2).mean()
    
    # Pixel-wise accuracy (within threshold)
    threshold = 0.05  # Within 5% of color range
    matches = np.abs(pred_np - truth_np) < threshold
    pixel_accuracy = matches.mean()
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "pixel_accuracy": float(pixel_accuracy),
        "exact_match": bool(matches.all()),
    }


def _comparison_canvas(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height))

    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width

    return canvas


def save_prediction_bundle(
    output_dir: str | Path,
    bundle_name: str,
    content_rgb: torch.Tensor,
    content_size: int,
    support_rgb: list[torch.Tensor],
    support_sizes: list[int],
    prediction_rgb: torch.Tensor,
    prediction_size: int,
    truth_rgb: torch.Tensor | None = None,
    truth_size: int | None = None,
    metadata: dict[str, str | int | float | bool | list[str]] | None = None,
    extra_metrics: dict[str, float | int | bool] | None = None,
) -> PredictionBundleResult:
    """Write a full sample/evaluation bundle to disk."""
    bundle_dir = Path(output_dir) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensors to PIL images
    def tensor_to_image(rgb_tensor: torch.Tensor, size: int) -> Image.Image:
        np_img = (rgb_tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(np_img)
        if size != IMAGE_SIZE:
            img = img.resize((size, size), Image.Resampling.NEAREST)
        return img

    content_image = tensor_to_image(content_rgb, content_size)
    support_images = [tensor_to_image(srgb, ssize) for srgb, ssize in zip(support_rgb, support_sizes)]
    prediction_image = tensor_to_image(prediction_rgb, prediction_size)

    original_path = bundle_dir / "original_tex.png"
    produced_path = bundle_dir / "produced_tex.png"
    content_image.save(original_path)
    prediction_image.save(produced_path)

    comparison_images = [content_image, prediction_image]
    truth_path: Path | None = None
    metrics: dict[str, float | int | bool] | None = None

    if truth_rgb is not None:
        resolved_truth_size = truth_size if truth_size is not None else prediction_size
        truth_image = tensor_to_image(truth_rgb, resolved_truth_size)
        truth_path = bundle_dir / "source_of_truth.png"
        truth_image.save(truth_path)
        comparison_images.append(truth_image)
        metrics = compute_metrics(prediction_rgb, truth_rgb)

    if extra_metrics:
        if metrics is None:
            metrics = {}
        metrics.update(extra_metrics)

    if metrics is not None:
        with open(bundle_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

    comparison_path = bundle_dir / "comparison.png"
    _comparison_canvas(comparison_images).save(comparison_path)

    if metadata is not None:
        with open(bundle_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    return {
        "bundle_dir": bundle_dir,
        "original_path": original_path,
        "support_path": bundle_dir / "support.png",
        "produced_path": produced_path,
        "truth_path": truth_path,
        "comparison_path": comparison_path,
        "metrics": metrics,
    }
