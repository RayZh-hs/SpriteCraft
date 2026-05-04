"""Training loop and utilities."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from spritecraft.config import CHECKPOINTS_DIR, IMAGE_SIZE, NUM_TIMESTEPS, VOCAB_SIZE
from spritecraft.data.dataset import TextureDataset
from spritecraft.inference.export import indices_to_image
from spritecraft.inference.sampler import sample_tokens
from spritecraft.models.diffusion import apply_mask
from spritecraft.models.unet import UNet


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "latest.pt"


def _save_checkpoint(
    checkpoint_path: Path,
    model: UNet,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    step: int,
    target_steps: int,
) -> None:
    torch.save(
        {
            "step": step,
            "target_steps": target_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        checkpoint_path,
    )


def _move_optimizer_state(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _maybe_load_checkpoint(
    checkpoint_path: Path,
    model: UNet,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: torch.device,
    target_steps: int,
) -> int:
    if not checkpoint_path.exists():
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    if checkpoint.get("target_steps") == target_steps:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("step", 0))


def _apply_cfg_dropout(content_ref: torch.Tensor, style_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    content_ref = content_ref.clone()
    style_ref = style_ref.clone()

    content_mask = torch.rand(content_ref.shape[0], device=content_ref.device) < 0.1
    style_mask = torch.rand(style_ref.shape[0], device=style_ref.device) < 0.1
    content_ref[content_mask] = 0
    style_ref[style_mask] = 0
    return content_ref, style_ref


def _make_grad_scaler(device: torch.device) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    enabled = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def _write_validation_preview(
    model: UNet,
    dataset: TextureDataset,
    checkpoint_dir: Path,
    step: int,
    device: torch.device,
) -> None:
    if len(dataset) == 0:
        return

    sample = dataset[step % len(dataset)]
    content_ref = sample["content_ref"].unsqueeze(0).to(device)
    style_ref = sample["style_ref"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)

    was_training = model.training
    model.eval()
    prediction = sample_tokens(model, content_ref, style_ref)
    if was_training:
        model.train()

    preview_dir = checkpoint_dir / "validation"
    preview_dir.mkdir(parents=True, exist_ok=True)

    content_image = indices_to_image(content_ref.squeeze(0).cpu().numpy())
    masked_image = Image.fromarray(np.full((IMAGE_SIZE, IMAGE_SIZE, 3), (255, 0, 255), dtype=np.uint8))
    prediction_image = indices_to_image(prediction.squeeze(0).cpu().numpy())
    target_image = indices_to_image(target.squeeze(0).cpu().numpy())

    canvas = Image.new("RGB", (content_image.width * 4, content_image.height))
    for tile_idx, tile in enumerate([content_image, masked_image, prediction_image, target_image]):
        canvas.paste(tile, (tile_idx * tile.width, 0))

    safe_filename = sample["filename"].replace(".png", "")
    preview_path = preview_dir / f"step_{step:06d}_{safe_filename}.png"
    canvas.save(preview_path)


def run(checkpoint_dir: str | Path = CHECKPOINTS_DIR, steps: int = 100_000):
    """Run training loop."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if steps <= 0:
        raise ValueError("steps must be positive")

    device = _select_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset = TextureDataset(split="train")
    val_dataset = TextureDataset(split="val")
    if len(train_dataset) == 0:
        raise ValueError("Training split is empty. Run preprocessing first.")

    batch_size = 2 if device.type == "cuda" else 1
    grad_accum_steps = 16 if device.type == "cuda" else 1
    validation_interval = 2_000
    save_interval = 500

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_iter = iter(train_loader)

    model = UNet(vocab_size=VOCAB_SIZE).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=1e-6)
    checkpoint_path = _latest_checkpoint_path(checkpoint_dir)
    start_step = _maybe_load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, steps)
    if start_step >= steps:
        print(f"Checkpoint already at step {start_step}, target was {steps}; nothing to do.")
        return checkpoint_path

    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    scaler = _make_grad_scaler(device)

    model.train()
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            content_ref = batch["content_ref"].to(device)
            style_ref = batch["style_ref"].to(device)
            target = batch["target"].to(device)

            content_ref, style_ref = _apply_cfg_dropout(content_ref, style_ref)
            t = torch.randint(1, NUM_TIMESTEPS + 1, (target.shape[0],), device=device)
            noisy_target = apply_mask(target, t)

            with autocast_context():
                logits = model(noisy_target, content_ref, style_ref, t)
                loss = F.cross_entropy(logits, target) / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += float(loss.detach().item()) * grad_accum_steps

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        current_step = step + 1
        if current_step == 1 or current_step % 10 == 0 or current_step == steps:
            lr = scheduler.get_last_lr()[0]
            print(f"step={current_step}/{steps} loss={total_loss / grad_accum_steps:.4f} lr={lr:.6e}")

        if current_step % save_interval == 0 or current_step == steps:
            _save_checkpoint(checkpoint_path, model, optimizer, scheduler, current_step, steps)
            step_checkpoint = checkpoint_dir / f"step_{current_step:06d}.pt"
            _save_checkpoint(step_checkpoint, model, optimizer, scheduler, current_step, steps)

        if current_step % validation_interval == 0:
            _write_validation_preview(model, val_dataset, checkpoint_dir, current_step, device)

    return checkpoint_path
