"""Training loop for per-pack RGB diffusion models."""

import csv
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from spritecraft.config import (
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    MAX_SUPPORT_EXEMPLARS,
    MIN_SUPPORT_EXEMPLARS,
    NUM_TIMESTEPS,
    pack_checkpoint_dir,
)
from spritecraft.data.dataset import PackStyleDataset, get_available_pack_ids
from spritecraft.debug.utility import (
    runtime_status_path,
    utcnow_iso,
    write_json_atomic,
)
from spritecraft.inference.sampler import sample_rgb
from spritecraft.models.diffusion import add_noise, get_alpha_schedule, get_beta_schedule, predict_x0_from_noise
from spritecraft.models.unet import StyleAwareUNet

MetricRecord = dict[str, Any]
METRIC_FIELDNAMES = ("step", "loss", "lr")
TENSORBOARD_DIRNAME = "tensorboard"


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "latest.pt"


def _tensorboard_log_dir(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / TENSORBOARD_DIRNAME


def _metrics_history_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "training_metrics.csv"


def _write_runtime_status(
    checkpoint_dir: Path,
    pid: int,
    step: int,
    target_steps: int,
    device: torch.device,
    batch_size: int,
    grad_accum_steps: int,
    validation_interval: int,
    save_interval: int,
    last_loss: float,
    last_lr: float,
    last_saved_step: int,
    last_validation_step: int,
    last_preview_dir: Path | None = None,
    status: str = "running",
) -> None:
    """Write runtime status JSON for debug tooling."""
    status_path = runtime_status_path(checkpoint_dir, pid)
    payload: dict[str, object] = {
        "pid": pid,
        "status": status,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "device": str(device),
        "step": step,
        "target_steps": target_steps,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "validation_interval": validation_interval,
        "save_interval": save_interval,
        "last_loss": last_loss,
        "last_lr": last_lr,
        "last_saved_step": last_saved_step,
        "last_validation_step": last_validation_step,
        "last_preview_dir": str(last_preview_dir.resolve()) if last_preview_dir is not None else None,
        "updated_at": utcnow_iso(),
    }
    write_json_atomic(status_path, payload)


def _write_metric_history(metrics_path: Path, metric_history: list[MetricRecord]) -> None:
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(cast(Any, metric_history))


def _append_metric_history(metrics_path: Path, metric_history: list[MetricRecord]) -> None:
    if not metric_history:
        return

    file_exists = metrics_path.exists() and metrics_path.stat().st_size > 0
    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(cast(Any, metric_history))


def _load_metric_history(metrics_path: Path, max_step: int) -> list[MetricRecord]:
    if not metrics_path.exists():
        return []

    records_by_step: dict[int, MetricRecord] = {}
    needs_rewrite = False
    previous_step = 0
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["step"])
            if step > max_step:
                needs_rewrite = True
                continue

            if step in records_by_step or step < previous_step:
                needs_rewrite = True

            records_by_step[step] = {
                "step": step,
                "loss": float(row["loss"]),
                "lr": float(row["lr"]),
            }
            previous_step = step

    metric_history = [records_by_step[step] for step in sorted(records_by_step)]
    if needs_rewrite:
        _write_metric_history(metrics_path, metric_history)
    return metric_history


def _save_checkpoint(
    checkpoint_path: Path,
    model: StyleAwareUNet,
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
    model: StyleAwareUNet,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: torch.device,
    target_steps: int,
) -> int:
    if not checkpoint_path.exists():
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with the current model. "
            "Use a fresh checkpoint directory after re-running preprocessing."
        ) from exc
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    if checkpoint.get("target_steps") == target_steps:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("step", 0))


def _make_grad_scaler(device: torch.device) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    enabled = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _luminance_gradient_map(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    luminance = (
        0.299 * rgb[:, 0:1] +
        0.587 * rgb[:, 1:2] +
        0.114 * rgb[:, 2:3]
    )
    grad_x = luminance[:, :, :, 1:] - luminance[:, :, :, :-1]
    grad_y = luminance[:, :, 1:, :] - luminance[:, :, :-1, :]
    return grad_x, grad_y


def _compute_loss(
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
    pred_x0: torch.Tensor,
    target_rgb: torch.Tensor,
) -> torch.Tensor:
    """Balance diffusion correctness with explicit edge preservation."""
    noise_loss = F.mse_loss(pred_noise, true_noise)
    recon_loss = F.l1_loss(pred_x0, target_rgb)
    pred_grad_x, pred_grad_y = _luminance_gradient_map(pred_x0)
    target_grad_x, target_grad_y = _luminance_gradient_map(target_rgb)
    gradient_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    return noise_loss + 0.5 * recon_loss + 0.25 * gradient_loss


def train_pack(
    pack_id: str,
    checkpoint_dir: Path,
    steps: int = 10_000,
    device: torch.device | None = None,
) -> Path:
    """Train a single per-pack model."""
    if device is None:
        device = _select_device()
    assert device is not None
    
    pack_checkpoint = pack_checkpoint_dir(checkpoint_dir, pack_id)
    pack_checkpoint.mkdir(parents=True, exist_ok=True)
    
    if steps <= 0:
        raise ValueError("steps must be positive")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load dataset
    train_dataset = PackStyleDataset(
        pack_id=pack_id,
        split="train",
        min_style_refs=MIN_SUPPORT_EXEMPLARS,
        max_style_refs=MAX_SUPPORT_EXEMPLARS,
    )
    val_dataset = PackStyleDataset(
        pack_id=pack_id,
        split="val",
        min_style_refs=MIN_SUPPORT_EXEMPLARS,
        max_style_refs=MAX_SUPPORT_EXEMPLARS,
    )
    
    if len(train_dataset) == 0:
        print(f"Skipping {pack_id}: no training samples")
        return pack_checkpoint

    batch_size = 4 if device.type == "cuda" else 1
    grad_accum_steps = 4 if device.type == "cuda" else 1
    validation_interval = 500
    save_interval = 500
    history_flush_interval = 10

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    train_iter = iter(train_loader)

    # Initialize model
    model = StyleAwareUNet(
        in_channels=3,
        style_channels=64,
        base_channels=128,
        num_style_refs=MAX_SUPPORT_EXEMPLARS,
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=1e-6)
    
    checkpoint_path = _latest_checkpoint_path(pack_checkpoint)
    start_step = _maybe_load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, steps)
    
    metrics_path = _metrics_history_path(pack_checkpoint)
    metric_history = _load_metric_history(metrics_path, max_step=start_step)
    pending_metric_history: list[MetricRecord] = []
    
    if start_step >= steps:
        print(f"[{pack_id}] Checkpoint already at step {start_step}, target was {steps}; nothing to do.")
        return checkpoint_path

    # Precompute diffusion schedule
    betas = get_beta_schedule(NUM_TIMESTEPS).to(device)
    alphas, alphas_cumprod = get_alpha_schedule(betas)
    alphas_cumprod = alphas_cumprod.to(device)

    tensorboard_dir = _tensorboard_log_dir(checkpoint_dir) / pack_id
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
    except ModuleNotFoundError:
        writer = None

    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    scaler = _make_grad_scaler(device)
    
    print(f"[{pack_id}] Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    print(f"[{pack_id}] Device: {device}, Batch size: {batch_size}, Steps: {steps}")

    pid = os.getpid()
    last_saved_step = start_step
    last_validation_step = start_step
    last_preview_dir: Path | None = None
    average_loss = float(metric_history[-1]["loss"]) if metric_history else 0.0
    lr = float(scheduler.get_last_lr()[0])

    model.train()
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for accum_idx in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            content_rgb = batch["content_rgb"].to(device)  # [B, 3, 32, 32]
            target_rgb = batch["target_rgb"].to(device)  # [B, 3, 32, 32]
            style_refs = batch["style_refs"].to(device)  # [B, N, 3, 32, 32]
            style_ref_mask = batch["style_ref_mask"].to(device)  # [B, N]

            # Sample timesteps
            t = torch.randint(1, NUM_TIMESTEPS + 1, (target_rgb.shape[0],), device=device)
            
            # Add noise
            noisy_target, true_noise = add_noise(target_rgb, t, alphas_cumprod)

            with autocast_context():
                pred_noise = model(noisy_target, content_rgb, style_refs, t, style_ref_mask=style_ref_mask)
                pred_x0 = torch.clamp(
                    predict_x0_from_noise(noisy_target, pred_noise, t, alphas_cumprod),
                    0.0,
                    1.0,
                )
                loss = _compute_loss(pred_noise, true_noise, pred_x0, target_rgb) / grad_accum_steps

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
        average_loss = total_loss / grad_accum_steps
        lr = float(scheduler.get_last_lr()[0])
        
        metric_record: MetricRecord = {
            "step": current_step,
            "loss": average_loss,
            "lr": lr,
        }
        metric_history.append(metric_record)
        pending_metric_history.append(metric_record)

        if writer is not None:
            writer.add_scalar("train/loss", average_loss, current_step)
            writer.add_scalar("train/learning_rate", lr, current_step)

        if current_step == 1 or current_step % history_flush_interval == 0 or current_step == steps:
            _append_metric_history(metrics_path, pending_metric_history)
            pending_metric_history.clear()
            if writer is not None:
                writer.flush()

        if current_step == 1 or current_step % 10 == 0 or current_step == steps:
            print(f"[{pack_id}] step={current_step}/{steps} loss={average_loss:.4f} lr={lr:.6e}")

        if current_step % save_interval == 0 or current_step == steps:
            if pending_metric_history:
                _append_metric_history(metrics_path, pending_metric_history)
                pending_metric_history.clear()
            _save_checkpoint(checkpoint_path, model, optimizer, scheduler, current_step, steps)
            step_checkpoint = pack_checkpoint / f"step_{current_step:06d}.pt"
            _save_checkpoint(step_checkpoint, model, optimizer, scheduler, current_step, steps)
            last_saved_step = current_step

        if current_step % validation_interval == 0 or current_step == steps:
            if len(val_dataset) > 0:
                preview_dir = _run_validation(model, val_dataset, pack_checkpoint, current_step, device, writer)
                if preview_dir is not None:
                    last_preview_dir = preview_dir
                    last_validation_step = current_step

        _write_runtime_status(
            checkpoint_dir=pack_checkpoint,
            pid=pid,
            step=current_step,
            target_steps=steps,
            device=device,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            validation_interval=validation_interval,
            save_interval=save_interval,
            last_loss=average_loss,
            last_lr=lr,
            last_saved_step=last_saved_step,
            last_validation_step=last_validation_step,
            last_preview_dir=last_preview_dir,
            status="running",
        )

    _write_runtime_status(
        checkpoint_dir=pack_checkpoint,
        pid=pid,
        step=steps,
        target_steps=steps,
        device=device,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        validation_interval=validation_interval,
        save_interval=save_interval,
        last_loss=average_loss,
        last_lr=lr,
        last_saved_step=last_saved_step,
        last_validation_step=last_validation_step,
        last_preview_dir=last_preview_dir,
        status="stopped",
    )

    if writer is not None:
        writer.close()
    
    return checkpoint_path


@torch.no_grad()
def _run_validation(
    model: StyleAwareUNet,
    val_dataset: PackStyleDataset,
    checkpoint_dir: Path,
    step: int,
    device: torch.device,
    writer: Any = None,
) -> Path | None:
    """Run validation and save preview images."""
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    preview_dir = checkpoint_dir / "validation" / f"step_{step:06d}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    total_loss = 0.0
    count = 0
    summary_images: list[Image.Image] = []
    
    for batch in val_loader:
        content_rgb = batch["content_rgb"].to(device)
        target_rgb = batch["target_rgb"].to(device)
        style_refs = batch["style_refs"].to(device)
        style_ref_mask = batch["style_ref_mask"].to(device)
        filename = batch["filename"][0]

        pred_rgb = sample_rgb(
            model,
            content_rgb,
            style_refs,
            style_ref_mask=style_ref_mask,
            num_steps=NUM_TIMESTEPS,
        ).unsqueeze(0).to(device)
        recon_loss = F.l1_loss(pred_rgb, target_rgb)
        pred_grad_x, pred_grad_y = _luminance_gradient_map(pred_rgb)
        target_grad_x, target_grad_y = _luminance_gradient_map(target_rgb)
        gradient_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        loss = recon_loss + 0.25 * gradient_loss
        total_loss += float(loss.item())
        count += 1
        
        # Save preview
        _save_preview(
            preview_dir,
            filename,
            content_rgb[0],
            target_rgb[0],
            pred_rgb[0],
            step,
        )
        
        # Collect images for tensorboard summary
        content_np = (content_rgb[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        target_np = (target_rgb[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        pred_np = (pred_rgb[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        canvas = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE))
        canvas.paste(Image.fromarray(content_np), (0, 0))
        canvas.paste(Image.fromarray(pred_np), (IMAGE_SIZE, 0))
        canvas.paste(Image.fromarray(target_np), (IMAGE_SIZE * 2, 0))
        summary_images.append(canvas)
    
    avg_loss = total_loss / max(count, 1)
    print(f"[{val_dataset.pack_id}] Validation step={step} loss={avg_loss:.4f}")
    
    # Log validation images to tensorboard
    if writer is not None:
        if summary_images:
            # Create a tiled summary image
            from spritecraft.inference.evaluate import _tile_images
            tiled = _tile_images(summary_images, columns=2)
            tiled_np = np.array(tiled)
            writer.add_image("validation/summary", tiled_np, step, dataformats="HWC")
    
    model.train()
    return preview_dir


def _save_preview(
    preview_dir: Path,
    filename: str,
    content_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    step: int,
) -> None:
    """Save a side-by-side preview image."""
    # Convert tensors to PIL images
    content_np = (content_rgb.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    target_np = (target_rgb.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    pred_np = (pred_rgb.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    
    content_img = Image.fromarray(content_np)
    target_img = Image.fromarray(target_np)
    pred_img = Image.fromarray(pred_np)
    
    # Create side-by-side canvas
    canvas = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE))
    canvas.paste(content_img, (0, 0))
    canvas.paste(pred_img, (IMAGE_SIZE, 0))
    canvas.paste(target_img, (IMAGE_SIZE * 2, 0))
    
    canvas.save(preview_dir / f"{filename}")


def run(checkpoint_dir: str | Path = CHECKPOINTS_DIR, steps: int = 10_000, pack_id: str | None = None):
    """Run training loop for one or all packs."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if pack_id is not None:
        # Train specific pack
        train_pack(pack_id, checkpoint_dir, steps)
    else:
        # Train all available packs
        available_packs = get_available_pack_ids()
        if not available_packs:
            raise ValueError("No preprocessed pack datasets found. Run preprocessing first.")
        
        print(f"Training {len(available_packs)} pack(s): {available_packs}")
        for pack_id in available_packs:
            try:
                train_pack(pack_id, checkpoint_dir, steps)
            except Exception as exc:
                print(f"Error training {pack_id}: {exc}")
                continue
