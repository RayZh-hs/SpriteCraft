"""Training loop for RecolorNet: structure-preserving color transfer."""

import csv
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from spritecraft.config import (
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    MAX_SUPPORT_EXEMPLARS,
    MIN_SUPPORT_EXEMPLARS,
    pack_checkpoint_dir,
)
from spritecraft.data.dataset import PackStyleDataset, get_available_pack_ids
from spritecraft.models.recolor import RecolorLoss, RecolorNet

LOSS_COMPONENT_NAMES = (
    "recon_loss",
    "edge_loss",
    "color_loss",
    "content_identity_loss",
)
TRAIN_SCALAR_NAMES = ("loss", "lr") + LOSS_COMPONENT_NAMES
METRIC_FIELDNAMES = ("step",) + TRAIN_SCALAR_NAMES
TENSORBOARD_DIRNAME = "tensorboard"

# Warmup steps as fraction of total training
WARMUP_FRACTION = 0.05
# EMA decay rate
EMA_DECAY = 0.995


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "recolor_latest.pt"


def _metrics_history_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "recolor_metrics.csv"


def _write_metric_history(metrics_path: Path, metric_history: list[dict[str, Any]]) -> None:
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        csv_writer.writeheader()
        csv_writer.writerows(cast(Any, metric_history))


def _save_checkpoint(
    checkpoint_path: Path,
    model: RecolorNet,
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
    model: RecolorNet,
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
            f"Checkpoint {checkpoint_path} is incompatible with the current model."
        ) from exc
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    if checkpoint.get("target_steps") == target_steps:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    ema_state_dict = checkpoint.get("ema_state_dict", None)
    if ema_state_dict is not None:
        ema_state_dict = {k: v.to(device) for k, v in ema_state_dict.items()}
    return int(checkpoint.get("step", 0)), ema_state_dict


def _update_ema(
    ema_state_dict: dict[str, torch.Tensor],
    model: RecolorNet,
    decay: float,
) -> dict[str, torch.Tensor]:
    """Update exponential moving average of model parameters."""
    device = next(model.parameters()).device
    if not ema_state_dict:
        ema_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        model_sd = model.state_dict()
        for key in ema_state_dict:
            ema_state_dict[key] = (
                decay * ema_state_dict[key].to(device) + (1 - decay) * model_sd[key].detach()
            )
    return ema_state_dict


def _apply_ema_weights(model: RecolorNet, ema_state_dict: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(ema_state_dict)


def _restore_training_weights(model: RecolorNet, training_state_dict: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(training_state_dict)


def _tensorboard_log_dir(checkpoint_dir: Path, pack_id: str) -> Path:
    return checkpoint_dir / TENSORBOARD_DIRNAME / pack_id


def train_recolor_pack(
    pack_id: str,
    checkpoint_dir: Path,
    steps: int = 5_000,
    device: torch.device | None = None,
    save_interval: int = 500,
    validation_interval: int = 500,
) -> Path:
    """Train a RecolorNet for one pack."""
    if device is None:
        device = _select_device()

    pack_checkpoint = pack_checkpoint_dir(checkpoint_dir, pack_id)
    pack_checkpoint.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
        print(f"[Recolor:{pack_id}] Skipping: no training samples")
        return pack_checkpoint

    batch_size = 8 if device.type == "cuda" else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_iter = iter(train_loader)

    model = RecolorNet(
        in_channels=3,
        style_channels=64,
        base_channels=64,
        num_style_refs=MAX_SUPPORT_EXEMPLARS,
    ).to(device)

    criterion = RecolorLoss(
        recon_weight=1.0,
        edge_weight=0.5,
        color_weight=1.5,
        content_identity_weight=0.05,
    )

    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    warmup_steps = max(1, int(steps * WARMUP_FRACTION))
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(steps - warmup_steps, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    checkpoint_path = _latest_checkpoint_path(pack_checkpoint)
    start_step, ema_state_dict = _maybe_load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, steps)

    metrics_path = _metrics_history_path(pack_checkpoint)

    if start_step >= steps:
        print(f"[Recolor:{pack_id}] Already at step {start_step}, target was {steps}")
        return checkpoint_path

    print(f"[Recolor:{pack_id}] Training on {len(train_dataset)} samples, {len(val_dataset)} val samples")
    print(f"[Recolor:{pack_id}] Device: {device}, Batch: {batch_size}, Steps: {steps}")

    tensorboard_dir = _tensorboard_log_dir(checkpoint_dir, pack_id) / "recolor"
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
    except ModuleNotFoundError:
        writer = None

    model.train()
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        content_rgb = batch["content_rgb"].to(device)
        target_rgb = batch["target_rgb"].to(device)
        style_refs = batch["style_refs"].to(device)
        style_ref_mask = batch["style_ref_mask"].to(device)

        pred = model(content_rgb, style_refs, style_ref_mask=style_ref_mask)
        loss_dict = criterion(pred, content_rgb, target_rgb)

        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update EMA after optimizer step
        ema_state_dict = _update_ema(ema_state_dict or {}, model, EMA_DECAY)

        current_step = step + 1
        lr = float(scheduler.get_last_lr()[0])

        if writer is not None:
            scalars = {k: float(v.item()) for k, v in loss_dict.items()}
            writer.add_scalar("training/0_summary/loss", scalars["loss"], current_step)
            writer.add_scalar("training/0_summary/learning_rate", lr, current_step)
            for name in LOSS_COMPONENT_NAMES:
                writer.add_scalar(f"training/1_details/{name}", scalars[name], current_step)

        if current_step == 1 or current_step % 50 == 0 or current_step == steps:
            scalars = {k: float(v.item()) for k, v in loss_dict.items()}
            print(
                f"[Recolor:{pack_id}] step={current_step}/{steps} "
                f"loss={scalars['loss']:.4f} "
                f"recon={scalars['recon_loss']:.4f} "
                f"color={scalars['color_loss']:.4f} "
                f"lr={lr:.6e}"
            )

        if current_step % save_interval == 0 or current_step == steps:
            _save_checkpoint(checkpoint_path, model, optimizer, scheduler, current_step, steps, ema_state_dict)

        if current_step % validation_interval == 0 or current_step == steps:
            _run_recolor_validation(model, val_dataset, pack_checkpoint, current_step, device, ema_state_dict, writer)

    if writer is not None:
        writer.close()

    return checkpoint_path


@torch.no_grad()
def _run_recolor_validation(
    model: RecolorNet,
    val_dataset: PackStyleDataset,
    checkpoint_dir: Path,
    step: int,
    device: torch.device,
    ema_state_dict: dict[str, torch.Tensor] | None = None,
    writer: Any = None,
) -> None:
    model.eval()

    training_state_dict = None
    if ema_state_dict:
        training_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        _apply_ema_weights(model, ema_state_dict)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    preview_dir = checkpoint_dir / "validation_recolor" / f"step_{step:06d}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    criterion = RecolorLoss(
        recon_weight=1.0,
        edge_weight=0.5,
        color_weight=1.5,
        content_identity_weight=0.05,
    )

    scalar_totals = {name: 0.0 for name in LOSS_COMPONENT_NAMES + ("loss",)}
    count = 0
    comparison_images: list[Image.Image] = []

    for batch in val_loader:
        content_rgb = batch["content_rgb"].to(device)
        target_rgb = batch["target_rgb"].to(device)
        style_refs = batch["style_refs"].to(device)
        style_ref_mask = batch["style_ref_mask"].to(device)
        filename = batch["filename"][0]

        pred = model(content_rgb, style_refs, style_ref_mask=style_ref_mask)
        loss_dict = criterion(pred, content_rgb, target_rgb)

        for name in scalar_totals:
            scalar_totals[name] += float(loss_dict[name].item())
        count += 1

        content_np = (content_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        target_np = (target_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pred_np = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

        canvas = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE))
        canvas.paste(Image.fromarray(content_np), (0, 0))
        canvas.paste(Image.fromarray(pred_np), (IMAGE_SIZE, 0))
        canvas.paste(Image.fromarray(target_np), (IMAGE_SIZE * 2, 0))
        canvas.save(preview_dir / filename)
        comparison_images.append(canvas)

    if count > 0:
        averaged = {k: v / count for k, v in scalar_totals.items()}
        print(f"[Recolor:{val_dataset.pack_id}] Validation step={step} loss={averaged['loss']:.4f}")

    if writer is not None:
        if count > 0:
            writer.add_scalar("validation/0_summary/loss", averaged["loss"], step)
            for name in LOSS_COMPONENT_NAMES:
                writer.add_scalar(f"validation/1_details/{name}", averaged[name], step)
        if comparison_images:
            from spritecraft.inference.evaluate import _tile_images
            tiled = _tile_images(comparison_images, columns=2)
            writer.add_image("validation/0_summary/recolor", np.array(tiled), step, dataformats="HWC")

    if training_state_dict is not None:
        _restore_training_weights(model, training_state_dict)
    model.train()


def run_recolor(checkpoint_dir: str | Path = CHECKPOINTS_DIR, steps: int = 5_000, pack_id: str | None = None):
    """Train RecolorNet for one or all packs."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if pack_id is not None:
        train_recolor_pack(pack_id, checkpoint_dir, steps)
    else:
        available_packs = get_available_pack_ids()
        if not available_packs:
            raise ValueError("No preprocessed pack datasets found.")
        print(f"Training RecolorNet for {len(available_packs)} pack(s): {available_packs}")
        for pid in available_packs:
            try:
                train_recolor_pack(pid, checkpoint_dir, steps)
            except Exception as exc:
                print(f"Error training RecolorNet for {pid}: {exc}")
                continue
