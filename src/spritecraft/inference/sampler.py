"""Iterative denoising sampler for continuous RGB diffusion."""

from contextlib import nullcontext
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image, ImageDraw

from spritecraft.config import IMAGE_SIZE, MAX_SUPPORT_EXEMPLARS, NUM_TIMESTEPS, pack_checkpoint_dir
from spritecraft.data.support_index import compute_texture_descriptor
from spritecraft.data.texture_classifier import evaluate_diffusion_output
from spritecraft.models.diffusion import ddpm_sample_step, get_alpha_schedule, get_beta_schedule
from spritecraft.models.recolor import RecolorNet
from spritecraft.models.unet import StyleAwareUNet

DETAIL_INJECTION_AMOUNTS = (0.35, 0.65)
# Minimum structural quality for diffusion to be considered viable.
# Below this, the output is visually garbled/smeared and should be rejected.
MIN_STRUCTURAL_QUALITY = 0.55
# Structural quality above which diffusion is confidently preferred over recolor.
GOOD_STRUCTURAL_QUALITY = 0.75


class PredictionBundleResult(TypedDict):
    bundle_dir: Path
    original_path: Path
    support_path: Path
    produced_path: Path
    truth_path: Path | None
    comparison_path: Path
    metrics: dict[str, float | int | bool] | None
    model_source: str
    model_score: float


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


def load_recolor_model(checkpoint_dir: str | Path, pack_id: str | None = None) -> tuple[RecolorNet, Path] | tuple[None, None]:
    """Load the latest RecolorNet checkpoint for a pack, if available."""
    checkpoint_dir = Path(checkpoint_dir)

    if pack_id is not None:
        checkpoint_dir = pack_checkpoint_dir(checkpoint_dir, pack_id)

    recolor_path = checkpoint_dir / "recolor_latest.pt"
    if not recolor_path.exists():
        return None, None

    checkpoint = torch.load(recolor_path, map_location="cpu")
    device = _select_device()
    model = RecolorNet(
        in_channels=3,
        style_channels=64,
        base_channels=64,
        num_style_refs=MAX_SUPPORT_EXEMPLARS,
    ).to(device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        print(f"Warning: RecolorNet checkpoint {recolor_path} is incompatible: {exc}")
        return None, None
    model.eval()
    return model, recolor_path


def _initial_sample(content_rgb: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Start the reverse process from the same pure-noise prior used in training."""
    del alphas_cumprod
    return torch.randn_like(content_rgb)


def _texture_descriptor(rgb_tensor: torch.Tensor) -> np.ndarray:
    """Convert a [3, H, W] RGB tensor in [0, 1] to a normalized descriptor."""
    image = (rgb_tensor.permute(1, 2, 0).detach().cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return compute_texture_descriptor(image)


def _sharpness_stats(rgb_tensor: torch.Tensor) -> tuple[float, float]:
    """Return simple edge/detail scalars for support-aware candidate ranking."""
    gray = (
        0.299 * rgb_tensor[0:1]
        + 0.587 * rgb_tensor[1:2]
        + 0.114 * rgb_tensor[2:3]
    )
    grad_x = gray[:, :, 1:] - gray[:, :, :-1]
    grad_y = gray[:, 1:, :] - gray[:, :-1, :]
    edge_energy = 0.5 * (grad_x.abs().mean().item() + grad_y.abs().mean().item())
    blurred = torch.nn.functional.avg_pool2d(gray.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    high_pass_energy = (gray - blurred).abs().mean().item()
    return edge_energy, high_pass_energy


def _active_reference_tensors(
    refs: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return unpadded reference tensors for a single sample."""
    if refs.dim() == 5:
        refs = refs[0]

    if style_ref_mask is not None:
        mask = style_ref_mask[0] if style_ref_mask.dim() == 2 else style_ref_mask
        refs = refs[mask.to(device=refs.device, dtype=torch.bool)]

    return refs


def _fit_channel_affine(
    source_refs: torch.Tensor,
    target_refs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-channel target ~= scale * source + bias from support pairs."""
    source_pixels = source_refs.permute(1, 0, 2, 3).reshape(3, -1)
    target_pixels = target_refs.permute(1, 0, 2, 3).reshape(3, -1)

    source_mean = source_pixels.mean(dim=1, keepdim=True)
    target_mean = target_pixels.mean(dim=1, keepdim=True)
    source_centered = source_pixels - source_mean
    target_centered = target_pixels - target_mean
    source_var = source_centered.square().mean(dim=1, keepdim=True)
    covariance = (source_centered * target_centered).mean(dim=1, keepdim=True)

    scale = (covariance / source_var.clamp_min(1e-5)).clamp(0.2, 3.0)
    bias = target_mean - scale * source_mean
    return scale.view(3, 1, 1), bias.view(3, 1, 1)


def _style_stat_recolor(
    content_rgb: torch.Tensor,
    style_refs: torch.Tensor,
) -> torch.Tensor:
    """Fallback recolor using support color moments when source support pairs are unavailable."""
    content_pixels = content_rgb.reshape(3, -1)
    style_pixels = style_refs.permute(1, 0, 2, 3).reshape(3, -1)

    content_mean = content_pixels.mean(dim=1, keepdim=True).view(3, 1, 1)
    content_std = content_pixels.std(dim=1, keepdim=True).view(3, 1, 1).clamp_min(1e-4)
    style_mean = style_pixels.mean(dim=1, keepdim=True).view(3, 1, 1)
    style_std = style_pixels.std(dim=1, keepdim=True).view(3, 1, 1)
    return ((content_rgb - content_mean) / content_std * style_std + style_mean).clamp(0.0, 1.0)


def _support_pair_recolor(
    content_rgb: torch.Tensor,
    style_refs: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
    support_content_refs: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Produce a vanilla-structure fallback by applying support-pair color migration."""
    content = content_rgb[0] if content_rgb.dim() == 4 else content_rgb
    active_style_refs = _active_reference_tensors(style_refs, style_ref_mask)
    if active_style_refs.shape[0] == 0:
        return None

    if support_content_refs is not None:
        active_source_refs = _active_reference_tensors(support_content_refs, style_ref_mask)
        if active_source_refs.shape[0] == active_style_refs.shape[0] and active_source_refs.shape[0] > 0:
            scale, bias = _fit_channel_affine(active_source_refs, active_style_refs)
            return (content * scale + bias).clamp(0.0, 1.0)

    return _style_stat_recolor(content, active_style_refs)


def _local_blur(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """Small edge-safe blur used to separate color mass from pixel detail."""
    padded = torch.nn.functional.pad(rgb_tensor.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    return torch.nn.functional.avg_pool2d(padded, kernel_size=3, stride=1).squeeze(0)


def _inject_detail(
    prediction_rgb: torch.Tensor,
    detail_source_rgb: torch.Tensor,
    amount: float,
) -> torch.Tensor:
    """Preserve generated color while borrowing high-frequency structure from a stable source."""
    prediction_low = _local_blur(prediction_rgb)
    source_detail = detail_source_rgb - _local_blur(detail_source_rgb)
    return (prediction_low + amount * source_detail).clamp(0.0, 1.0)


def _support_descriptor_score(
    prediction_rgb: torch.Tensor,
    style_refs: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
) -> float:
    """Score a sample by support similarity while rejecting under-sharp candidates."""
    prediction_descriptor = _texture_descriptor(prediction_rgb)
    prediction_edge, prediction_high_pass = _sharpness_stats(prediction_rgb)

    if style_refs.dim() == 5:
        refs = style_refs[0]
    else:
        refs = style_refs

    if style_ref_mask is not None:
        if style_ref_mask.dim() == 2:
            keep = style_ref_mask[0]
        else:
            keep = style_ref_mask
        refs = refs[keep]

    if refs.shape[0] == 0:
        return float(np.linalg.norm(prediction_descriptor))

    support_descriptors = [_texture_descriptor(ref_rgb) for ref_rgb in refs]
    support_stats = [_sharpness_stats(ref_rgb) for ref_rgb in refs]
    support_mean = np.mean(np.stack(support_descriptors, axis=0), axis=0)
    support_mean = support_mean / (np.linalg.norm(support_mean) + 1e-6)
    support_edge = float(np.mean([edge for edge, _high_pass in support_stats]))
    support_high_pass = float(np.mean([high_pass for _edge, high_pass in support_stats]))
    prediction_descriptor = prediction_descriptor / (np.linalg.norm(prediction_descriptor) + 1e-6)
    descriptor_score = float(np.dot(prediction_descriptor, support_mean))

    # Penalize candidates that are noticeably softer than the support set.
    # This improves stochastic candidate selection without incentivizing
    # arbitrarily noisy or over-sharpened outputs.
    edge_shortfall = max(0.0, support_edge - prediction_edge) / (support_edge + 1e-6)
    high_pass_shortfall = max(0.0, support_high_pass - prediction_high_pass) / (support_high_pass + 1e-6)
    sharpness_penalty = 0.2 * edge_shortfall + 0.1 * high_pass_shortfall
    return descriptor_score - sharpness_penalty


@torch.no_grad()
def _sample_once(
    model: StyleAwareUNet,
    content_rgb: torch.Tensor,
    style_refs: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
    num_steps: int = NUM_TIMESTEPS,
) -> torch.Tensor:
    """Run a single stochastic reverse process and return a CPU RGB tensor."""
    device = next(model.parameters()).device

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


@torch.no_grad()
def sample_rgb(
    model: StyleAwareUNet,
    content_rgb: torch.Tensor,
    style_refs: torch.Tensor,
    style_ref_mask: torch.Tensor | None = None,
    support_content_refs: torch.Tensor | None = None,
    recolor_model: RecolorNet | None = None,
    num_steps: int = NUM_TIMESTEPS,
    num_candidates: int = 1,
    return_source: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, str, float]:
    """Generate RGB texture. Runs diffusion, evaluates output quality against
    content structure AND target style, compares with RecolorNet, and selects
    the best output. No hardcoded routing — the decision is made by comparing
    actual output quality of both models.

    Args:
        model: trained StyleAwareUNet
        content_rgb: [1, 3, H, W] or [3, H, W] vanilla content
        style_refs: [N, 3, H, W] or [1, N, 3, H, W] style reference textures
        support_content_refs: optional vanilla-side matches for each support
        recolor_model: optional trained RecolorNet for fallback color transfer
        num_steps: number of denoising steps
        num_candidates: number of stochastic diffusion candidates
        return_source: if True, return (prediction, model_source, score)

    Returns:
        [3, H, W] generated RGB texture, or (prediction, model_source, score)
    """
    if content_rgb.dim() == 3:
        content_rgb = content_rgb.unsqueeze(0)
    if style_refs.dim() == 4:
        style_refs = style_refs.unsqueeze(0)
    if style_ref_mask is not None and style_ref_mask.dim() == 1:
        style_ref_mask = style_ref_mask.unsqueeze(0)
    if support_content_refs is not None and support_content_refs.dim() == 4:
        support_content_refs = support_content_refs.unsqueeze(0)

    device = next(model.parameters()).device
    content_rgb = content_rgb.to(device)
    style_refs = style_refs.to(device)
    if style_ref_mask is not None:
        style_ref_mask = style_ref_mask.to(device)
    if support_content_refs is not None:
        support_content_refs = support_content_refs.to(device)

    # Precompute deterministic recolor (cheap)
    recolor_candidate = _support_pair_recolor(
        content_rgb,
        style_refs,
        style_ref_mask=style_ref_mask,
        support_content_refs=support_content_refs,
    )

    # Helper: compute structural quality of a candidate vs content.
    # prediction must be [3, H, W] CPU float tensor in [0, 1].
    content_np = (content_rgb.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)

    def _structural_quality(prediction: torch.Tensor) -> float:
        pred_np = (prediction.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        return evaluate_diffusion_output(pred_np, content_np)["quality_score"]

    def _style_quality(prediction: torch.Tensor) -> float:
        score = _support_descriptor_score(prediction, style_refs, style_ref_mask=style_ref_mask)
        return max(0.0, min(1.0, score))

    # === PHASE 1: Generate diffusion candidates ===
    effective_candidates = max(1, num_candidates)
    best_diffusion_pred = None
    best_diffusion_style = float("-inf")
    best_diffusion_struct = 0.0

    for _ in range(effective_candidates):
        prediction = _sample_once(
            model,
            content_rgb,
            style_refs,
            style_ref_mask=style_ref_mask,
            num_steps=num_steps,
        )
        struct = _structural_quality(prediction)
        style = _style_quality(prediction)
        if style > best_diffusion_style or best_diffusion_pred is None:
            best_diffusion_pred = prediction
            best_diffusion_style = style
            best_diffusion_struct = struct

    assert best_diffusion_pred is not None

    # Try detail injection to improve structural quality
    best_pred = best_diffusion_pred
    best_struct = best_diffusion_struct
    best_style = best_diffusion_style
    best_source = "diffusion"

    if recolor_candidate is not None:
        for amount in DETAIL_INJECTION_AMOUNTS:
            injected = _inject_detail(
                best_diffusion_pred.to(device), recolor_candidate, amount
            ).cpu()
            injected_struct = _structural_quality(injected)
            injected_style = _style_quality(injected)
            if injected_struct > best_struct and injected_style > 0.95 * best_style:
                best_pred = injected
                best_struct = injected_struct
                best_style = injected_style
                best_source = f"diffusion+detail_{amount:.2f}"

    # Determine diffusion viability: structural quality must meet minimum bar
    diffusion_viable = best_struct >= MIN_STRUCTURAL_QUALITY
    diffusion_confident = best_struct >= GOOD_STRUCTURAL_QUALITY

    # === PHASE 2: Generate RecolorNet candidate ===
    recolor_style = float("-inf")
    recolor_pred = None

    if recolor_model is not None:
        recolor_rgb = recolor_model(
            content_rgb, style_refs, style_ref_mask=style_ref_mask
        ).squeeze(0)
        recolor_pred = recolor_rgb.cpu()
        recolor_style = _style_quality(recolor_pred)

    # Decision: when diffusion has confident structure, prefer it unless
    # recolor is clearly better on style. When diffusion is borderline,
    # prefer the structurally perfect recolor unless diffusion is clearly
    # better on style. When diffusion is garbage, always fall back.
    CLEAR_MARGIN = 0.10

    if recolor_pred is None:
        result = best_pred
        chosen_source = best_source
        chosen_score = best_style
    elif diffusion_confident:
        # Diffusion has strong structure — prefer it unless recolor clearly wins
        if recolor_style > best_style + CLEAR_MARGIN:
            result = recolor_pred
            chosen_source = "recolor_net"
            chosen_score = recolor_style
        else:
            result = best_pred
            chosen_source = best_source
            chosen_score = best_style
    elif diffusion_viable:
        # Borderline structure — prefer recolor unless diffusion clearly wins
        if best_style > recolor_style + CLEAR_MARGIN:
            result = best_pred
            chosen_source = best_source
            chosen_score = best_style
        else:
            result = recolor_pred
            chosen_source = "recolor_net"
            chosen_score = recolor_style
    else:
        # Diffusion structurally poor — fall back to recolor
        result = recolor_pred
        chosen_source = "recolor_net"
        chosen_score = recolor_style

    if return_source:
        return result, chosen_source, float(chosen_score)
    return result


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
    model_source: str = "unknown",
    model_score: float = 0.0,
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
        metrics["model_source"] = model_source
        metrics["model_score"] = model_score
        with open(bundle_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

    comparison_path = bundle_dir / "comparison.png"
    # Annotate comparison with model source
    comparison_labels[1] = f"generated [{model_source}]"
    _comparison_canvas(comparison_images, labels=comparison_labels).save(comparison_path)

    if metadata is not None:
        metadata["model_source"] = model_source
        metadata["model_score"] = model_score
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
        "model_source": model_source,
        "model_score": model_score,
    }
