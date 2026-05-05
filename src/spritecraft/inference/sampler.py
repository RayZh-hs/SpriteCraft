"""Iterative decoding sampler."""

from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image

from spritecraft.config import (
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    MASK_TOKEN,
    NUM_TIMESTEPS,
    OUTPUT_DIR,
    PALETTE_PATH,
    VOCAB_SIZE,
)
from spritecraft.data.preprocess import preprocess_image, quantize_image
from spritecraft.inference.export import indices_to_image
from spritecraft.models.unet import UNet


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
    support_content_refs: torch.Tensor,
    support_style_refs: torch.Tensor,
    support_mask: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    conditional_logits = model(
        noisy_target,
        content_ref,
        support_content_refs,
        support_style_refs,
        support_mask,
        t,
    )
    if guidance_scale == 1.0:
        return conditional_logits

    null_content = torch.zeros_like(content_ref)
    null_support_content = torch.zeros_like(support_content_refs)
    null_support_style = torch.zeros_like(support_style_refs)
    null_support_mask = torch.zeros_like(support_mask)
    unconditional_logits = model(
        noisy_target,
        null_content,
        null_support_content,
        null_support_style,
        null_support_mask,
        t,
    )
    return unconditional_logits + guidance_scale * (conditional_logits - unconditional_logits)


def load_model(checkpoint_dir: str | Path = CHECKPOINTS_DIR) -> tuple[UNet, Path]:
    """Load the latest checkpoint from a directory or a specific checkpoint path."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir if checkpoint_dir.is_file() else _latest_checkpoint_path(checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = _select_device()
    model = UNet(vocab_size=VOCAB_SIZE).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with the current support-pair model. "
            "Train into a fresh checkpoint directory after re-running preprocessing."
        ) from exc
    model.eval()
    return model, checkpoint_path


@torch.no_grad()
def sample_tokens(
    model: UNet,
    content_ref: torch.Tensor,
    support_content_refs: torch.Tensor,
    support_style_refs: torch.Tensor,
    support_mask: torch.Tensor | None = None,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    device = next(model.parameters()).device
    if content_ref.dim() == 2:
        content_ref = content_ref.unsqueeze(0)
    if support_content_refs.dim() == 3:
        support_content_refs = support_content_refs.unsqueeze(0)
    if support_style_refs.dim() == 3:
        support_style_refs = support_style_refs.unsqueeze(0)

    content_ref = content_ref.to(device)
    support_content_refs = support_content_refs.to(device)
    support_style_refs = support_style_refs.to(device)
    if support_mask is None:
        support_mask = torch.ones(support_content_refs.shape[:2], device=device, dtype=torch.bool)
    elif support_mask.dim() == 1:
        support_mask = support_mask.unsqueeze(0)
    support_mask = support_mask.to(device=device, dtype=torch.bool)
    noisy_target = torch.full_like(content_ref, fill_value=MASK_TOKEN, device=device)

    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )

    prediction: torch.Tensor | None = None
    for timestep in range(NUM_TIMESTEPS, 0, -1):
        if not noisy_target.eq(MASK_TOKEN).any():
            break

        t = torch.full((noisy_target.shape[0],), timestep, dtype=torch.long, device=device)
        with autocast_context():
            logits = _guided_logits(
                model,
                noisy_target,
                content_ref,
                support_content_refs,
                support_style_refs,
                support_mask,
                t,
                guidance_scale,
            )

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
        if prediction is None:
            raise RuntimeError("Sampler exited without a prediction tensor.")
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


def _support_pairs_canvas(
    support_original_images: list[Image.Image],
    support_styled_images: list[Image.Image],
) -> Image.Image:
    if len(support_original_images) != len(support_styled_images):
        raise ValueError("Support original/styled image counts must match")

    paired_canvases = [
        _comparison_canvas([original_image, styled_image])
        for original_image, styled_image in zip(support_original_images, support_styled_images)
    ]
    return _comparison_canvas(paired_canvases)


def save_prediction_bundle(
    output_dir: str | Path,
    bundle_name: str,
    content_tokens: torch.Tensor,
    content_size: int,
    support_content_tokens: list[torch.Tensor],
    support_content_sizes: list[int],
    support_style_tokens: list[torch.Tensor],
    support_style_sizes: list[int],
    prediction_tokens: torch.Tensor,
    prediction_size: int,
    truth_tokens: torch.Tensor | None = None,
    truth_size: int | None = None,
    metadata: dict[str, str | int | float | bool | list[str]] | None = None,
    extra_metrics: dict[str, float | int | bool] | None = None,
) -> PredictionBundleResult:
    """Write a full sample/evaluation bundle to disk."""
    bundle_dir = Path(output_dir) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    content_image = indices_to_image(content_tokens.cpu().numpy(), target_size=content_size)
    support_original_images = [
        indices_to_image(tokens.cpu().numpy(), target_size=size)
        for tokens, size in zip(support_content_tokens, support_content_sizes)
    ]
    support_styled_images = [
        indices_to_image(tokens.cpu().numpy(), target_size=size)
        for tokens, size in zip(support_style_tokens, support_style_sizes)
    ]
    support_pairs_image = _support_pairs_canvas(support_original_images, support_styled_images)
    prediction_image = indices_to_image(prediction_tokens.cpu().numpy(), target_size=prediction_size)

    original_path = bundle_dir / "original_tex.png"
    support_path = bundle_dir / "support_pairs.png"
    produced_path = bundle_dir / "produced_tex.png"
    content_image.save(original_path)
    support_pairs_image.save(support_path)
    prediction_image.save(produced_path)

    comparison_images = [content_image, support_pairs_image, prediction_image]
    truth_path: Path | None = None
    metrics: dict[str, float | int | bool] | None = None

    if truth_tokens is not None:
        resolved_truth_size = truth_size if truth_size is not None else prediction_size
        truth_image = indices_to_image(truth_tokens.cpu().numpy(), target_size=resolved_truth_size)
        truth_path = bundle_dir / "source_of_truth.png"
        truth_image.save(truth_path)
        comparison_images.append(truth_image)
        metrics = compute_metrics(prediction_tokens, truth_tokens)

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
        "support_path": support_path,
        "produced_path": produced_path,
        "truth_path": truth_path,
        "comparison_path": comparison_path,
        "metrics": metrics,
    }


def run(
    content_path: str,
    support_original_paths: list[str],
    support_styled_paths: list[str],
    output_dir: str | Path = OUTPUT_DIR,
    checkpoint_dir: str | Path = CHECKPOINTS_DIR,
    truth_path: str | None = None,
):
    """Generate a texture from a vanilla target and support exemplar pairs."""
    if not support_original_paths or not support_styled_paths:
        raise ValueError("At least one support pair is required")
    if len(support_original_paths) != len(support_styled_paths):
        raise ValueError("support_original_paths and support_styled_paths must have the same length")

    palette = np.load(PALETTE_PATH)
    content_ref, content_size = _load_reference_tokens(Path(content_path), palette)

    support_original_refs: list[torch.Tensor] = []
    support_original_sizes: list[int] = []
    support_styled_refs: list[torch.Tensor] = []
    support_styled_sizes: list[int] = []
    for support_original_path, support_styled_path in zip(support_original_paths, support_styled_paths):
        support_original_ref, support_original_size = _load_reference_tokens(Path(support_original_path), palette)
        support_styled_ref, support_styled_size = _load_reference_tokens(Path(support_styled_path), palette)
        support_original_refs.append(support_original_ref)
        support_original_sizes.append(support_original_size)
        support_styled_refs.append(support_styled_ref)
        support_styled_sizes.append(support_styled_size)

    truth_ref: torch.Tensor | None = None
    truth_size: int | None = None
    if truth_path is not None:
        truth_ref, truth_size = _load_reference_tokens(Path(truth_path), palette)

    model, checkpoint_path = load_model(checkpoint_dir)
    generated = sample_tokens(
        model,
        content_ref.unsqueeze(0),
        torch.stack(support_original_refs, dim=0),
        torch.stack(support_styled_refs, dim=0),
    ).squeeze(0)

    bundle_name = f"{Path(content_path).stem}_from_{len(support_original_paths)}_support_pairs"
    prediction_size = truth_size if truth_size is not None else content_size
    metadata: dict[str, str | int | float | bool | list[str]] = {
        "content_path": str(Path(content_path).resolve()),
        "support_original_paths": [str(Path(path).resolve()) for path in support_original_paths],
        "support_styled_paths": [str(Path(path).resolve()) for path in support_styled_paths],
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    if truth_path is not None:
        metadata["truth_path"] = str(Path(truth_path).resolve())

    result = save_prediction_bundle(
        output_dir=output_dir,
        bundle_name=bundle_name,
        content_tokens=content_ref,
        content_size=content_size,
        support_content_tokens=support_original_refs,
        support_content_sizes=support_original_sizes,
        support_style_tokens=support_styled_refs,
        support_style_sizes=support_styled_sizes,
        prediction_tokens=generated,
        prediction_size=prediction_size,
        truth_tokens=truth_ref,
        truth_size=truth_size,
        metadata=metadata,
    )

    print(f"Saved original texture to {result['original_path']}")
    print(f"Saved support pairs to {result['support_path']}")
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
