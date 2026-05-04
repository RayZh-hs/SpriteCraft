"""Iterative decoding sampler."""

from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spritecraft.config import CHECKPOINTS_DIR, IMAGE_SIZE, MASK_TOKEN, NUM_TIMESTEPS, OUTPUT_DIR, PALETTE_PATH, VOCAB_SIZE
from spritecraft.data.preprocess import preprocess_image, quantize_image
from spritecraft.inference.export import indices_to_image
from spritecraft.models.unet import UNet


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

    nested_latest = list(checkpoint_dir.glob("**/latest.pt"))
    if nested_latest:
        return max(nested_latest, key=lambda path: path.stat().st_mtime)

    nested_steps = sorted(checkpoint_dir.glob("**/step_*.pt"))
    if nested_steps:
        return nested_steps[-1]

    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def _load_reference_tokens(image_path: Path, palette: np.ndarray) -> tuple[torch.Tensor, int]:
    with Image.open(image_path) as img:
        img.load()
        original_width, original_height = img.size

        if original_width != original_height:
            raise ValueError(f"Expected square texture, got {original_width}x{original_height} from {image_path}")

        processed = preprocess_image(img)
        if processed.size != (IMAGE_SIZE, IMAGE_SIZE):
            processed = processed.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)

    token_indices = quantize_image(processed, palette)
    output_size = 16 if original_width == 16 else IMAGE_SIZE
    return torch.as_tensor(token_indices, dtype=torch.long), output_size


def _guided_logits(
    model: UNet,
    noisy_target: torch.Tensor,
    content_ref: torch.Tensor,
    style_ref: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    conditional_logits = model(noisy_target, content_ref, style_ref, t)
    if guidance_scale == 1.0:
        return conditional_logits

    null_content = torch.zeros_like(content_ref)
    null_style = torch.zeros_like(style_ref)
    unconditional_logits = model(noisy_target, null_content, null_style, t)
    return unconditional_logits + guidance_scale * (conditional_logits - unconditional_logits)


def load_model(checkpoint_dir: str | Path = CHECKPOINTS_DIR) -> tuple[UNet, Path]:
    """Load the latest available checkpoint and return an eval-ready model."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = _latest_checkpoint_path(checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = _select_device()
    model = UNet(vocab_size=VOCAB_SIZE).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint_path


@torch.no_grad()
def sample_tokens(
    model: UNet,
    content_ref: torch.Tensor,
    style_ref: torch.Tensor,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    device = next(model.parameters()).device
    content_ref = content_ref.to(device)
    style_ref = style_ref.to(device)
    noisy_target = torch.full_like(content_ref, fill_value=MASK_TOKEN, device=device)

    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )

    for timestep in range(NUM_TIMESTEPS, 0, -1):
        if not noisy_target.eq(MASK_TOKEN).any():
            break

        t = torch.full((noisy_target.shape[0],), timestep, dtype=torch.long, device=device)
        with autocast_context():
            logits = _guided_logits(model, noisy_target, content_ref, style_ref, t, guidance_scale)

        probabilities = torch.softmax(logits.float(), dim=1)
        confidence, prediction = probabilities.max(dim=1)
        masked_positions = noisy_target.eq(MASK_TOKEN)

        for batch_idx in range(noisy_target.shape[0]):
            flat_mask = masked_positions[batch_idx].view(-1)
            remaining = int(flat_mask.sum().item())
            if remaining == 0:
                continue

            reveal_count = max(1, math.ceil(remaining / timestep))
            masked_indices = flat_mask.nonzero(as_tuple=False).squeeze(1)
            flat_confidence = confidence[batch_idx].view(-1)
            flat_prediction = prediction[batch_idx].view(-1)

            if reveal_count >= remaining:
                chosen_indices = masked_indices
            else:
                masked_confidence = flat_confidence[masked_indices]
                topk = torch.topk(masked_confidence, k=reveal_count).indices
                chosen_indices = masked_indices[topk]

            flat_target = noisy_target[batch_idx].view(-1)
            flat_target[chosen_indices] = flat_prediction[chosen_indices]

    if noisy_target.eq(MASK_TOKEN).any():
        noisy_target = torch.where(noisy_target.eq(MASK_TOKEN), prediction, noisy_target)

    return noisy_target.cpu()


def compute_metrics(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float | int | bool]:
    """Compute simple texture metrics against a ground-truth texture."""
    prediction_array = prediction.cpu().numpy()
    truth_array = truth.cpu().numpy()

    matches = prediction_array == truth_array
    prediction_rgb = np.asarray(indices_to_image(prediction_array), dtype=np.int16)
    truth_rgb = np.asarray(indices_to_image(truth_array), dtype=np.int16)

    return {
        "matching_pixels": int(matches.sum()),
        "total_pixels": int(matches.size),
        "pixel_accuracy": float(matches.mean()),
        "exact_match": bool(matches.all()),
        "rgb_mae": float(np.abs(prediction_rgb - truth_rgb).mean()),
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
    content_tokens: torch.Tensor,
    content_size: int,
    style_tokens: torch.Tensor,
    style_size: int,
    prediction_tokens: torch.Tensor,
    prediction_size: int,
    truth_tokens: torch.Tensor | None = None,
    truth_size: int | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> dict[str, Path | dict[str, float | int | bool] | None]:
    """Write a full sample/evaluation bundle to disk."""
    bundle_dir = Path(output_dir) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    content_image = indices_to_image(content_tokens.cpu().numpy(), target_size=content_size)
    style_image = indices_to_image(style_tokens.cpu().numpy(), target_size=style_size)
    prediction_image = indices_to_image(prediction_tokens.cpu().numpy(), target_size=prediction_size)

    original_path = bundle_dir / "original_tex.png"
    style_path = bundle_dir / "style_reference.png"
    produced_path = bundle_dir / "produced_tex.png"
    content_image.save(original_path)
    style_image.save(style_path)
    prediction_image.save(produced_path)

    comparison_images = [content_image, style_image, prediction_image]
    truth_path: Path | None = None
    metrics: dict[str, float | int | bool] | None = None

    if truth_tokens is not None:
        resolved_truth_size = truth_size if truth_size is not None else prediction_size
        truth_image = indices_to_image(truth_tokens.cpu().numpy(), target_size=resolved_truth_size)
        truth_path = bundle_dir / "source_of_truth.png"
        truth_image.save(truth_path)
        comparison_images.append(truth_image)
        metrics = compute_metrics(prediction_tokens, truth_tokens)
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
        "style_path": style_path,
        "produced_path": produced_path,
        "truth_path": truth_path,
        "comparison_path": comparison_path,
        "metrics": metrics,
    }


def run(
    content_path: str,
    style_path: str,
    output_dir: str = OUTPUT_DIR,
    checkpoint_dir: str | Path = CHECKPOINTS_DIR,
    truth_path: str | None = None,
):
    """Generate a texture from content and style references."""
    palette = np.load(PALETTE_PATH)
    content_ref, content_size = _load_reference_tokens(Path(content_path), palette)
    style_ref, style_size = _load_reference_tokens(Path(style_path), palette)
    truth_ref: torch.Tensor | None = None
    truth_size: int | None = None
    if truth_path is not None:
        truth_ref, truth_size = _load_reference_tokens(Path(truth_path), palette)

    model, checkpoint_path = load_model(checkpoint_dir)
    generated = sample_tokens(model, content_ref.unsqueeze(0), style_ref.unsqueeze(0)).squeeze(0)

    bundle_name = f"{Path(content_path).stem}_styled_like_{Path(style_path).stem}"
    prediction_size = truth_size if truth_size is not None else style_size
    metadata = {
        "content_path": str(Path(content_path).resolve()),
        "style_path": str(Path(style_path).resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    if truth_path is not None:
        metadata["truth_path"] = str(Path(truth_path).resolve())

    result = save_prediction_bundle(
        output_dir=output_dir,
        bundle_name=bundle_name,
        content_tokens=content_ref,
        content_size=content_size,
        style_tokens=style_ref,
        style_size=style_size,
        prediction_tokens=generated,
        prediction_size=prediction_size,
        truth_tokens=truth_ref,
        truth_size=truth_size,
        metadata=metadata,
    )

    print(f"Saved original texture to {result['original_path']}")
    print(f"Saved style reference to {result['style_path']}")
    print(f"Saved generated texture to {result['produced_path']}")
    if result["truth_path"] is not None:
        print(f"Saved source of truth to {result['truth_path']}")
    print(f"Saved side-by-side comparison to {result['comparison_path']}")

    metrics = result["metrics"]
    if metrics is not None:
        print(
            "Metrics: "
            f"pixel_accuracy={metrics['pixel_accuracy']:.4f} "
            f"rgb_mae={metrics['rgb_mae']:.2f} "
            f"exact_match={metrics['exact_match']}"
        )

    return result["bundle_dir"]
