"""Iterative decoding sampler."""

from contextlib import nullcontext
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


def run(
    content_path: str,
    style_path: str,
    output_dir: str = OUTPUT_DIR,
    checkpoint_dir: str | Path = CHECKPOINTS_DIR,
):
    """Generate a texture from content and style references."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(checkpoint_dir)

    palette = np.load(PALETTE_PATH)
    content_ref, _ = _load_reference_tokens(Path(content_path), palette)
    style_ref, output_size = _load_reference_tokens(Path(style_path), palette)

    checkpoint_path = _latest_checkpoint_path(checkpoint_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = _select_device()
    model = UNet(vocab_size=VOCAB_SIZE).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generated = sample_tokens(model, content_ref.unsqueeze(0), style_ref.unsqueeze(0))
    image = indices_to_image(generated.squeeze(0).numpy(), target_size=output_size)

    output_path = output_dir / f"{Path(content_path).stem}_styled_like_{Path(style_path).stem}.png"
    image.save(output_path)
    print(f"Saved generated texture to {output_path}")
    return output_path
