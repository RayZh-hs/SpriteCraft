"""Iterative denoising sampler for continuous RGB diffusion."""

from contextlib import nullcontext
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image, ImageDraw

from spritecraft.config import (
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    MAX_SUPPORT_EXEMPLARS,
    NUM_TIMESTEPS,
    OUTPUT_DIR,
    pack_checkpoint_dir,
)
from spritecraft.models.diffusion import ddpm_sample_step, get_alpha_schedule, get_beta_schedule
from spritecraft.models.unet import StyleAwareUNet

CONTENT_INIT_MIN_ALPHA_CUMPROD = 0.25


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
        num_style_refs=MAX_SUPPORT_EXEMPLARS,
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


def _initial_sample(content_rgb: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Choose a reverse-process start state that matches the training regime.

    A short diffusion schedule with a weak terminal noise level never teaches the
    model to recover from pure Gaussian noise. In that case, start from the
    terminal forward-diffused content image instead of an unconditional noise
    sample so existing checkpoints remain usable.
    """
    terminal_alpha_cumprod = float(alphas_cumprod[-1].item())
    if terminal_alpha_cumprod >= CONTENT_INIT_MIN_ALPHA_CUMPROD:
        alpha = torch.sqrt(alphas_cumprod[-1]).view(1, 1, 1, 1)
        sigma = torch.sqrt(1 - alphas_cumprod[-1]).view(1, 1, 1, 1)
        return alpha * content_rgb + sigma * torch.randn_like(content_rgb)
    return torch.randn_like(content_rgb)


@torch.no_grad()
def sample_rgb(
    model: StyleAwareUNet,
    content_rgb: torch.Tensor,
    style_refs: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
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
    if style_ref_mask is not None and style_ref_mask.dim() == 1:
        style_ref_mask = style_ref_mask.unsqueeze(0)
    
    content_rgb = content_rgb.to(device)
    style_refs = style_refs.to(device)
    if style_ref_mask is not None:
        style_ref_mask = style_ref_mask.to(device)
    
    # Precompute diffusion schedule
    betas = get_beta_schedule(num_steps).to(device)
    alphas, alphas_cumprod = get_alpha_schedule(betas)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    
    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )

    x = _initial_sample(content_rgb, alphas_cumprod)
    
    # Iterative denoising
    for timestep in range(num_steps, 0, -1):
        t = torch.full((content_rgb.shape[0],), timestep, dtype=torch.long, device=device)
        
        with autocast_context():
            pred_noise = model(x, content_rgb, style_refs, t, style_ref_mask=style_ref_mask)

        x = ddpm_sample_step(x, pred_noise, t, betas, alphas, alphas_cumprod, clip_x0=True)

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


def _comparison_canvas(
    images: list[Image.Image],
    labels: list[str] | None = None,
    gap: int = 2,
    label_height: int = 10,
) -> Image.Image:
    width = sum(image.width for image in images) + gap * max(len(images) - 1, 0)
    height = max(image.height for image in images)
    if labels is not None:
        height += label_height
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x_offset = 0
    for index, image in enumerate(images):
        image_y = label_height if labels is not None else 0
        canvas.paste(image, (x_offset, image_y))
        if labels is not None and index < len(labels):
            draw.text((x_offset + 1, 0), labels[index], fill=(0, 0, 0))
        x_offset += image.width
        if index < len(images) - 1:
            draw.rectangle(
                [(x_offset, 0), (x_offset + gap - 1, height - 1)],
                fill=(235, 235, 235),
            )
            x_offset += gap

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
    support_path = bundle_dir / "support.png"
    produced_path = bundle_dir / "produced_tex.png"
    content_image.save(original_path)
    prediction_image.save(produced_path)
    if support_images:
        _comparison_canvas(
            support_images,
            labels=[f"support_{index + 1}" for index in range(len(support_images))],
        ).save(support_path)
    else:
        Image.new("RGB", (1, 1), color=(255, 255, 255)).save(support_path)

    comparison_images = [content_image, prediction_image]
    comparison_labels = ["vanilla", "generated"]
    truth_path: Path | None = None
    metrics: dict[str, float | int | bool] | None = None

    if truth_rgb is not None:
        resolved_truth_size = truth_size if truth_size is not None else prediction_size
        truth_image = tensor_to_image(truth_rgb, resolved_truth_size)
        truth_path = bundle_dir / "source_of_truth.png"
        truth_image.save(truth_path)
        comparison_images.append(truth_image)
        comparison_labels.append("truth")
        metrics = compute_metrics(prediction_rgb, truth_rgb)

    if extra_metrics:
        if metrics is None:
            metrics = {}
        metrics.update(extra_metrics)

    if metrics is not None:
        with open(bundle_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

    comparison_path = bundle_dir / "comparison.png"
    _comparison_canvas(comparison_images, labels=comparison_labels).save(comparison_path)

    if metadata is not None:
        with open(bundle_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    return {
        "bundle_dir": bundle_dir,
        "original_path": original_path,
        "support_path": support_path,
        "produced_path": produced_path,
        "truth_path": truth_path,
        "comparison_path": comparison_path,
        "metrics": metrics,
    }
