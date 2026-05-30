"""Training loop for per-pack RGB diffusion models."""

import csv
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any, cast
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from spritecraft.config import (
    CHECKPOINTS_DIR,
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
from spritecraft.inference.sampler import _sample_once, load_recolor_model, sample_rgb
from spritecraft.models.diffusion import add_noise, get_alpha_schedule, get_beta_schedule, predict_x0_from_noise
from spritecraft.models.recolor import RecolorNet
from spritecraft.models.unet import StyleAwareUNet
from spritecraft.training.content_loss import rgb_content_loss_components

MetricRecord = dict[str, Any]
LOSS_COMPONENT_NAMES = (
    "noise_loss",
    "recon_loss",
    "gradient_loss",
    "content_loss",
    "content_structure_loss",
    "content_gradient_delta_loss",
    "content_detail_delta_loss",
    "content_contrast_loss",
    "content_hue_loss",
    "content_color_moment_loss",
)
TRAIN_SCALAR_NAMES = ("loss", "lr") + LOSS_COMPONENT_NAMES
VAL_SCALAR_NAMES = (
    "loss",
    "recon_loss",
    "gradient_loss",
    "content_loss",
    "content_structure_loss",
    "content_gradient_delta_loss",
    "content_detail_delta_loss",
    "content_contrast_loss",
    "content_hue_loss",
    "content_color_moment_loss",
)
METRIC_FIELDNAMES = ("step",) + TRAIN_SCALAR_NAMES
TENSORBOARD_DIRNAME = "tensorboard"

# Warmup steps as fraction of total training
WARMUP_FRACTION = 0.05
# Content loss ramp: start at this fraction, reach full weight at this step fraction
CONTENT_LOSS_RAMP_START = 0.1
CONTENT_LOSS_RAMP_END = 0.2  # reach full weight by 20% of training
# EMA decay rate
EMA_DECAY = 0.995


def _stable_validation_seed(filename: str) -> int:
    """Return a stable per-texture seed so validation previews are comparable."""
    return zlib.crc32(filename.encode("utf-8")) & 0x7FFFFFFF


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
        csv_writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        csv_writer.writeheader()
        csv_writer.writerows(cast(Any, metric_history))


def _append_metric_history(metrics_path: Path, metric_history: list[MetricRecord]) -> None:
    if not metric_history:
        return

    file_exists = metrics_path.exists() and metrics_path.stat().st_size > 0
    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        if not file_exists:
            csv_writer.writeheader()
        csv_writer.writerows(cast(Any, metric_history))


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
                "loss": float(row.get("loss", 0.0)),
                "lr": float(row.get("lr", 0.0)),
                "noise_loss": float(row.get("noise_loss", 0.0)),
                "recon_loss": float(row.get("recon_loss", 0.0)),
                "gradient_loss": float(row.get("gradient_loss", 0.0)),
                "content_loss": float(row.get("content_loss", 0.0)),
                "content_structure_loss": float(row.get("content_structure_loss", 0.0)),
                "content_gradient_delta_loss": float(row.get("content_gradient_delta_loss", 0.0)),
                "content_detail_delta_loss": float(row.get("content_detail_delta_loss", 0.0)),
                "content_contrast_loss": float(row.get("content_contrast_loss", 0.0)),
                "content_hue_loss": float(row.get("content_hue_loss", 0.0)),
                "content_color_moment_loss": float(row.get("content_color_moment_loss", 0.0)),
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
    scheduler: SequentialLR | CosineAnnealingLR,
    step: int,
    target_steps: int,
    ema_state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    checkpoint_data: dict[str, Any] = {
        "step": step,
        "target_steps": target_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if ema_state_dict is not None:
        checkpoint_data["ema_state_dict"] = ema_state_dict
    torch.save(checkpoint_data, checkpoint_path)


def _move_optimizer_state(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _maybe_load_checkpoint(
    checkpoint_path: Path,
    model: StyleAwareUNet,
    optimizer: AdamW,
    scheduler: SequentialLR | CosineAnnealingLR,
    device: torch.device,
    target_steps: int,
) -> tuple[int, dict[str, torch.Tensor] | None]:
    if not checkpoint_path.exists():
        return 0, None

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
    ema_state_dict = checkpoint.get("ema_state_dict", None)
    if ema_state_dict is not None:
        ema_state_dict = {k: v.to(device) for k, v in ema_state_dict.items()}
    return int(checkpoint.get("step", 0)), ema_state_dict


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


def _rgb_channel_gradient_loss(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    """Per-channel gradient loss that penalizes color bleeding and blurring."""
    total = torch.tensor(0.0, device=pred_rgb.device)
    for c in range(3):
        channel_pred = pred_rgb[:, c:c+1]
        channel_target = target_rgb[:, c:c+1]
        gx_pred = channel_pred[:, :, :, 1:] - channel_pred[:, :, :, :-1]
        gy_pred = channel_pred[:, :, 1:, :] - channel_pred[:, :, :-1, :]
        gx_target = channel_target[:, :, :, 1:] - channel_target[:, :, :, :-1]
        gy_target = channel_target[:, :, 1:, :] - channel_target[:, :, :-1, :]
        total = total + F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)
    return total / 3.0


def _update_ema(
    ema_state_dict: dict[str, torch.Tensor],
    model: StyleAwareUNet,
    decay: float,
) -> dict[str, torch.Tensor]:
    """Update exponential moving average of model parameters."""
    device = next(model.parameters()).device
    if not ema_state_dict:
        ema_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        model_sd = model.state_dict()
        for key in ema_state_dict:
            ema_state_dict[key] = decay * ema_state_dict[key].to(device) + (1 - decay) * model_sd[key].detach()
    return ema_state_dict


def _apply_ema_weights(model: StyleAwareUNet, ema_state_dict: dict[str, torch.Tensor]) -> None:
    """Temporarily swap model weights with EMA weights for validation."""
    model.load_state_dict(ema_state_dict)


def _restore_training_weights(model: StyleAwareUNet, training_state_dict: dict[str, torch.Tensor]) -> None:
    """Restore training weights after EMA validation."""
    model.load_state_dict(training_state_dict)


def _content_loss_weight_for_step(step: int, total_steps: int) -> float:
    """Ramp content loss weight from 0.10 to 0.30 over the first CONTENT_LOSS_RAMP_END fraction."""
    ramp_start = int(total_steps * CONTENT_LOSS_RAMP_START)
    ramp_end = int(total_steps * CONTENT_LOSS_RAMP_END)
    if step < ramp_start:
        return 0.10
    elif step >= ramp_end:
        return 0.30
    else:
        progress = (step - ramp_start) / max(ramp_end - ramp_start, 1)
        return 0.10 + (0.30 - 0.10) * progress


def _compute_loss(
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
    pred_x0: torch.Tensor,
    content_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    content_loss_weight: float = 0.30,
) -> dict[str, torch.Tensor]:
    """Balance diffusion correctness with explicit edge and color preservation."""
    noise_loss = F.mse_loss(pred_noise, true_noise)
    recon_loss = F.l1_loss(pred_x0, target_rgb)
    pred_grad_x, pred_grad_y = _luminance_gradient_map(pred_x0)
    target_grad_x, target_grad_y = _luminance_gradient_map(target_rgb)
    luminance_gradient_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    channel_gradient_loss = _rgb_channel_gradient_loss(pred_x0, target_rgb)
    gradient_loss = luminance_gradient_loss + 0.5 * channel_gradient_loss
    components = rgb_content_loss_components(pred_x0, content_rgb, target_rgb)
    total_loss = noise_loss + 0.70 * recon_loss + 0.30 * gradient_loss + content_loss_weight * components["content_loss"]
    return {
        "loss": total_loss,
        "noise_loss": noise_loss,
        "recon_loss": recon_loss,
        "gradient_loss": gradient_loss,
        **components,
    }


def _init_scalar_totals(names: tuple[str, ...]) -> dict[str, float]:
    return {name: 0.0 for name in names}


def _average_scalar_totals(totals: dict[str, float], divisor: float) -> dict[str, float]:
    if divisor <= 0:
        return {name: 0.0 for name in totals}
    return {name: value / divisor for name, value in totals.items()}


def _build_output_panel(
    content_rgb: torch.Tensor,
    output_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    label: str,
    border_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    def _to_img(t: torch.Tensor) -> Image.Image:
        arr = (t.detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return Image.fromarray(arr)

    content_img = _to_img(content_rgb)
    output_img = _to_img(output_rgb)
    target_img = _to_img(target_rgb)

    label_height = 12
    gap = 2
    img_size = content_img.width
    canvas_w = 3 * img_size + 2 * gap
    canvas_h = img_size + label_height
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((1, 0), "content", fill=(0, 0, 0))
    canvas.paste(content_img, (0, label_height))

    x_out = img_size + gap
    draw.text((x_out + 1, 0), label, fill=(0, 0, 0))
    canvas.paste(output_img, (x_out, label_height))

    x_target = 2 * (img_size + gap)
    draw.text((x_target + 1, 0), "target", fill=(0, 0, 0))
    canvas.paste(target_img, (x_target, label_height))

    if border_color is not None:
        for b in range(3):
            draw.rectangle(
                [(x_out - b, label_height - b),
                 (x_out + img_size - 1 + b, label_height + img_size - 1 + b)],
                outline=border_color,
            )

    return canvas


def train_pack(
    pack_id: str,
    checkpoint_dir: Path,
    steps: int = 10_000,
    device: torch.device | None = None,
    recolor_checkpoint_dir: Path | None = None,
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
    warmup_steps = max(1, int(steps * WARMUP_FRACTION))
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(steps - warmup_steps, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    checkpoint_path = _latest_checkpoint_path(pack_checkpoint)
    start_step, ema_state_dict = _maybe_load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, steps)
    
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

    tensorboard_dir = _tensorboard_log_dir(checkpoint_dir) / pack_id / "std"
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
        scalar_totals = _init_scalar_totals(LOSS_COMPONENT_NAMES + ("loss",))

        current_step = step + 1
        content_loss_weight = _content_loss_weight_for_step(step, steps)

        for _accum_idx in range(grad_accum_steps):
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
                loss_components = _compute_loss(
                    pred_noise, true_noise, pred_x0, content_rgb, target_rgb,
                    content_loss_weight=content_loss_weight,
                )
                loss = loss_components["loss"] / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            for name in scalar_totals:
                scalar_totals[name] += float(loss_components[name].detach().item())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Update EMA after optimizer step
        ema_state_dict = _update_ema(ema_state_dict or {}, model, EMA_DECAY)

        averaged_scalars = _average_scalar_totals(scalar_totals, grad_accum_steps)
        average_loss = averaged_scalars["loss"]
        lr = float(scheduler.get_last_lr()[0])
        
        metric_record: MetricRecord = {
            "step": current_step,
            "loss": average_loss,
            "lr": lr,
            **{name: averaged_scalars[name] for name in LOSS_COMPONENT_NAMES},
        }
        metric_history.append(metric_record)
        pending_metric_history.append(metric_record)

        if writer is not None:
            writer.add_scalar("training (summary)/loss", average_loss, current_step)
            writer.add_scalar("training (summary)/learning_rate", lr, current_step)
            for name in LOSS_COMPONENT_NAMES:
                writer.add_scalar(f"training (details)/{name}", averaged_scalars[name], current_step)

        if current_step == 1 or current_step % history_flush_interval == 0 or current_step == steps:
            _append_metric_history(metrics_path, pending_metric_history)
            pending_metric_history.clear()
            if writer is not None:
                writer.flush()

        if current_step == 1 or current_step % 10 == 0 or current_step == steps:
            print(
                f"[{pack_id}] step={current_step}/{steps} "
                f"loss={average_loss:.4f} "
                f"recon={averaged_scalars['recon_loss']:.4f} "
                f"content={averaged_scalars['content_loss']:.4f} "
                f"contrast={averaged_scalars['content_contrast_loss']:.4f} "
                f"lr={lr:.6e}"
            )

        if current_step % save_interval == 0 or current_step == steps:
            if pending_metric_history:
                _append_metric_history(metrics_path, pending_metric_history)
                pending_metric_history.clear()
            _save_checkpoint(checkpoint_path, model, optimizer, scheduler, current_step, steps, ema_state_dict)
            step_checkpoint = pack_checkpoint / f"step_{current_step:06d}.pt"
            _save_checkpoint(step_checkpoint, model, optimizer, scheduler, current_step, steps, ema_state_dict)
            last_saved_step = current_step

        if current_step % validation_interval == 0 or current_step == steps:
            if len(val_dataset) > 0:
                preview_dir = _run_validation(
                    model, val_dataset, pack_checkpoint, current_step, device, writer,
                    ema_state_dict=ema_state_dict,
                    recolor_checkpoint_dir=recolor_checkpoint_dir,
                )
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
    ema_state_dict: dict[str, torch.Tensor] | None = None,
    recolor_checkpoint_dir: Path | None = None,
) -> Path | None:
    """Run validation and save preview images."""
    model.eval()

    training_state_dict = None
    if ema_state_dict:
        training_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        _apply_ema_weights(model, ema_state_dict)

    recolor_model: RecolorNet | None = None
    if recolor_checkpoint_dir is not None:
        recolor_model, _ = load_recolor_model(recolor_checkpoint_dir, val_dataset.pack_id)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    preview_dir = checkpoint_dir / "validation" / f"step_{step:06d}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    scalar_totals = _init_scalar_totals(VAL_SCALAR_NAMES)
    count = 0
    recolor_images: list[Image.Image] = []
    diffusion_images: list[Image.Image] = []
    selected_images: list[Image.Image] = []
    loss_breakdown_rows: list[dict[str, float | str]] = []

    for batch in val_loader:
        content_rgb = batch["content_rgb"].to(device)
        target_rgb = batch["target_rgb"].to(device)
        style_refs = batch["style_refs"].to(device)
        style_ref_mask = batch["style_ref_mask"].to(device)
        support_content_refs = batch["support_content_refs"].to(device)
        filename = batch["filename"][0]

        validation_seed = _stable_validation_seed(filename)
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(validation_seed)

        pure_recolor: torch.Tensor | None = None
        if recolor_model is not None:
            pure_recolor = recolor_model(
                content_rgb, style_refs, style_ref_mask=style_ref_mask
            ).squeeze(0)

        pure_diff = _sample_once(
            model, content_rgb, style_refs, style_ref_mask=style_ref_mask
        )

        selected, chosen_source, chosen_score = sample_rgb(
            model,
            content_rgb,
            style_refs,
            style_ref_mask=style_ref_mask,
            support_content_refs=support_content_refs,
            recolor_model=recolor_model,
            num_steps=NUM_TIMESTEPS,
            num_candidates=2,
            return_source=True,
        )

        selected = selected.unsqueeze(0).to(device)
        pred_rgb = selected

        content_components = rgb_content_loss_components(pred_rgb, content_rgb, target_rgb)
        recon_loss = F.l1_loss(pred_rgb, target_rgb)
        pred_grad_x, pred_grad_y = _luminance_gradient_map(pred_rgb)
        target_grad_x, target_grad_y = _luminance_gradient_map(target_rgb)
        luminance_gradient_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        channel_gradient_loss = _rgb_channel_gradient_loss(pred_rgb, target_rgb)
        gradient_loss = luminance_gradient_loss + 0.5 * channel_gradient_loss
        loss = recon_loss + 0.30 * gradient_loss + 0.30 * content_components["content_loss"]
        sample_scalars = {
            "loss": float(loss.item()),
            "recon_loss": float(recon_loss.item()),
            "gradient_loss": float(gradient_loss.item()),
            "content_loss": float(content_components["content_loss"].item()),
            "content_structure_loss": float(content_components["content_structure_loss"].item()),
            "content_gradient_delta_loss": float(content_components["content_gradient_delta_loss"].item()),
            "content_detail_delta_loss": float(content_components["content_detail_delta_loss"].item()),
            "content_contrast_loss": float(content_components["content_contrast_loss"].item()),
            "content_hue_loss": float(content_components["content_hue_loss"].item()),
            "content_color_moment_loss": float(content_components["content_color_moment_loss"].item()),
        }
        for name in VAL_SCALAR_NAMES:
            scalar_totals[name] += sample_scalars[name]
        count += 1
        loss_breakdown_rows.append({"filename": filename, **sample_scalars})

        stem = Path(filename).stem
        is_diffusion = "diffusion" in chosen_source
        is_recolor = "recolor" in chosen_source

        if pure_recolor is not None:
            recolor_panel = _build_output_panel(
                content_rgb[0], pure_recolor, target_rgb[0], "recolor",
            )
            recolor_panel.save(preview_dir / f"{stem}_recolor.png")
            recolor_images.append(recolor_panel)

        diff_panel = _build_output_panel(
            content_rgb[0], pure_diff, target_rgb[0], "diffusion",
        )
        diff_panel.save(preview_dir / f"{stem}_diffusion.png")
        diffusion_images.append(diff_panel)

        sel_border = None
        if is_diffusion:
            sel_border = (34, 170, 34)
        elif is_recolor:
            sel_border = (34, 68, 204)
        sel_panel = _build_output_panel(
            content_rgb[0], selected[0], target_rgb[0],
            f"selected [{chosen_source}]", border_color=sel_border,
        )
        sel_panel.save(preview_dir / f"{stem}_selected.png")
        selected_images.append(sel_panel)

    averaged_scalars = _average_scalar_totals(scalar_totals, count)
    avg_loss = averaged_scalars["loss"]
    print(f"[{val_dataset.pack_id}] Validation step={step} loss={avg_loss:.4f}")

    breakdown_path = preview_dir / "loss_breakdown.csv"
    with breakdown_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=("filename",) + VAL_SCALAR_NAMES)
        csv_writer.writeheader()
        csv_writer.writerows(cast(Any, loss_breakdown_rows))

    if writer is not None:
        from spritecraft.inference.evaluate import _tile_images
        writer.add_scalar("validation (summary)/loss", avg_loss, step)
        for name in VAL_SCALAR_NAMES:
            if name != "loss":
                writer.add_scalar(f"validation (details)/{name}", averaged_scalars[name], step)
        if recolor_images:
            tiled = _tile_images(recolor_images, columns=2)
            writer.add_image("validation (summary)/recolor", np.array(tiled), step, dataformats="HWC")
        if diffusion_images:
            tiled = _tile_images(diffusion_images, columns=2)
            writer.add_image("validation (summary)/diffusion", np.array(tiled), step, dataformats="HWC")
        if selected_images:
            tiled = _tile_images(selected_images, columns=2)
            writer.add_image("validation (summary)/selected", np.array(tiled), step, dataformats="HWC")

    if training_state_dict is not None:
        _restore_training_weights(model, training_state_dict)

    model.train()
    return preview_dir


def run(checkpoint_dir: str | Path = CHECKPOINTS_DIR, steps: int = 10_000, pack_id: str | None = None, recolor_checkpoint_dir: str | Path | None = None):
    """Run training loop for one or all packs."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    recolor_cp_dir = Path(recolor_checkpoint_dir) if recolor_checkpoint_dir is not None else None
    
    if pack_id is not None:
        train_pack(pack_id, checkpoint_dir, steps, recolor_checkpoint_dir=recolor_cp_dir)
    else:
        available_packs = get_available_pack_ids()
        if not available_packs:
            raise ValueError("No preprocessed pack datasets found. Run preprocessing first.")
        
        print(f"Training {len(available_packs)} pack(s): {available_packs}")
        for pack_id in available_packs:
            try:
                train_pack(pack_id, checkpoint_dir, steps, recolor_checkpoint_dir=recolor_cp_dir)
            except Exception as exc:
                print(f"Error training {pack_id}: {exc}")
                continue
