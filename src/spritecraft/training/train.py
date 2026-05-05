"""Training loop and utilities."""

import csv
from contextlib import nullcontext
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from spritecraft.config import (
    CHECKPOINTS_DIR,
    MASK_TOKEN,
    MAX_SUPPORT_EXEMPLARS,
    MIN_SUPPORT_EXEMPLARS,
    NUM_TIMESTEPS,
    VOCAB_SIZE,
)
from spritecraft.data.dataset import TextureDataset
from spritecraft.inference.evaluate import write_validation_matrix
from spritecraft.models.diffusion import apply_mask
from spritecraft.models.unet import UNet

MetricRecord = dict[str, float | int]
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


def _validation_preview_dir(checkpoint_dir: Path, step: int) -> Path:
    return checkpoint_dir / "validation" / f"step_{step:06d}"


def _make_summary_writer(log_dir: Path, start_step: int):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard support requires the 'tensorboard' package. "
            "Reinstall project dependencies before training."
        ) from exc

    log_dir.mkdir(parents=True, exist_ok=True)
    purge_step = start_step + 1 if start_step > 0 else None
    writer_kwargs = {"log_dir": str(log_dir)}
    if purge_step is not None:
        writer_kwargs["purge_step"] = purge_step
    return SummaryWriter(**writer_kwargs)


def _collate_training_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    """Stack tensor payloads while preserving variable-length sample metadata."""
    collated: dict[str, object] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if torch.is_tensor(values[0]):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    return collated


def _write_metric_history(metrics_path: Path, metric_history: list[MetricRecord]) -> None:
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(metric_history)


def _append_metric_history(metrics_path: Path, metric_history: list[MetricRecord]) -> None:
    if not metric_history:
        return

    file_exists = metrics_path.exists() and metrics_path.stat().st_size > 0
    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(metric_history)


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


def _format_metric_value(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return f"{value:.2e}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.4f}"


def _plot_points(
    values: list[float],
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[list[tuple[int, int]], float, float]:
    if not values:
        return [], 0.0, 1.0

    y_min = min(values)
    y_max = max(values)
    span = y_max - y_min
    if math.isclose(span, 0.0):
        padding = max(abs(y_min) * 0.05, 1e-6)
    else:
        padding = span * 0.1

    plot_y_min = y_min - padding
    plot_y_max = y_max + padding
    if y_min >= 0:
        plot_y_min = max(0.0, plot_y_min)
    if math.isclose(plot_y_min, plot_y_max):
        plot_y_max = plot_y_min + 1.0

    denominator = len(values) - 1
    points: list[tuple[int, int]] = []
    for index, value in enumerate(values):
        if denominator <= 0:
            x = left + width // 2
        else:
            x = left + round(width * index / denominator)
        y_ratio = (value - plot_y_min) / (plot_y_max - plot_y_min)
        y = top + height - round(y_ratio * height)
        points.append((x, y))

    return points, plot_y_min, plot_y_max


def _write_metric_graph(
    metric_history: list[MetricRecord],
    metric_key: str,
    title: str,
    output_path: Path,
    line_color: tuple[int, int, int],
) -> None:
    if not metric_history:
        return

    width, height = 960, 540
    margin_left, margin_top = 88, 56
    margin_right, margin_bottom = 28, 56
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    font = ImageFont.load_default()

    steps = [int(record["step"]) for record in metric_history]
    values = [float(record[metric_key]) for record in metric_history]
    latest_value = values[-1]

    image = Image.new("RGB", (width, height), color=(250, 251, 252))
    draw = ImageDraw.Draw(image)

    points, plot_y_min, plot_y_max = _plot_points(values, margin_left, margin_top, plot_width, plot_height)

    for tick_idx in range(5):
        tick_ratio = tick_idx / 4
        y = margin_top + round(plot_height * tick_ratio)
        tick_value = plot_y_max - (plot_y_max - plot_y_min) * tick_ratio
        draw.line(
            [(margin_left, y), (margin_left + plot_width, y)],
            fill=(226, 232, 240),
            width=1,
        )
        draw.text((12, y - 6), _format_metric_value(tick_value), fill=(71, 85, 105), font=font)

    step_min = steps[0]
    step_max = steps[-1]
    for tick_idx in range(5):
        tick_ratio = tick_idx / 4
        x = margin_left + round(plot_width * tick_ratio)
        step_value = step_min if step_max == step_min else round(step_min + (step_max - step_min) * tick_ratio)
        draw.line(
            [(x, margin_top), (x, margin_top + plot_height)],
            fill=(238, 242, 247),
            width=1,
        )
        draw.text((x - 10, margin_top + plot_height + 14), str(step_value), fill=(71, 85, 105), font=font)

    draw.rectangle(
        [(margin_left, margin_top), (margin_left + plot_width, margin_top + plot_height)],
        outline=(148, 163, 184),
        width=1,
    )
    if len(points) > 1:
        draw.line(points, fill=line_color, width=3)
    last_x, last_y = points[-1]
    draw.ellipse(
        [(last_x - 4, last_y - 4), (last_x + 4, last_y + 4)],
        fill=line_color,
        outline=(255, 255, 255),
        width=1,
    )

    draw.text((margin_left, 18), title, fill=(15, 23, 42), font=font)
    summary = (
        f"latest={_format_metric_value(latest_value)}  "
        f"min={_format_metric_value(min(values))}  "
        f"max={_format_metric_value(max(values))}  "
        f"points={len(values)}"
    )
    draw.text((margin_left, 34), summary, fill=(71, 85, 105), font=font)
    draw.text((width // 2 - 18, height - 24), "step", fill=(51, 65, 85), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_training_graphs(checkpoint_dir: Path, metric_history: list[MetricRecord]) -> None:
    _write_metric_graph(
        metric_history,
        metric_key="loss",
        title="Training Loss",
        output_path=checkpoint_dir / "training_loss.png",
        line_color=(37, 99, 235),
    )
    _write_metric_graph(
        metric_history,
        metric_key="lr",
        title="Learning Rate",
        output_path=checkpoint_dir / "training_learning_rate.png",
        line_color=(22, 163, 74),
    )


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
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is incompatible with the current support-pair model. "
            "Use a fresh checkpoint directory after re-running preprocessing."
        ) from exc
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    if checkpoint.get("target_steps") == target_steps:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("step", 0))


def _apply_cfg_dropout(
    content_ref: torch.Tensor,
    support_content_refs: torch.Tensor,
    support_style_refs: torch.Tensor,
    support_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    content_ref = content_ref.clone()
    support_content_refs = support_content_refs.clone()
    support_style_refs = support_style_refs.clone()
    support_mask = support_mask.clone()

    content_mask = torch.rand(content_ref.shape[0], device=content_ref.device) < 0.1
    support_dropout_mask = torch.rand(support_content_refs.shape[0], device=support_content_refs.device) < 0.1
    content_ref[content_mask] = 0
    support_content_refs[support_dropout_mask] = 0
    support_style_refs[support_dropout_mask] = 0
    support_mask[support_dropout_mask] = False
    return content_ref, support_content_refs, support_style_refs, support_mask


def _make_grad_scaler(device: torch.device) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    enabled = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _parameter_norm(model: UNet) -> float:
    squared_norm = 0.0
    for parameter in model.parameters():
        squared_norm += float(parameter.detach().float().pow(2).sum().item())
    return math.sqrt(squared_norm)


def _gradient_norm(model: UNet) -> float:
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        squared_norm += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(squared_norm)


def _sanitize_tag_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _collect_batch_diagnostics(
    logits: torch.Tensor,
    target: torch.Tensor,
    noisy_target: torch.Tensor,
    support_mask: torch.Tensor,
    attention_weights: torch.Tensor | None,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    probabilities = torch.softmax(logits.detach().float(), dim=1)
    confidence, prediction = probabilities.max(dim=1)
    flat_prediction = prediction.reshape(prediction.shape[0], -1)

    unique_counts = torch.tensor(
        [torch.unique(sample).numel() for sample in flat_prediction],
        device=flat_prediction.device,
        dtype=torch.float32,
    )
    dominant_token_shares = torch.stack(
        [
            torch.bincount(sample, minlength=VOCAB_SIZE - 1).max().float() / sample.numel()
            for sample in flat_prediction
        ]
    )

    scalar_metrics = {
        "train/token_accuracy": float(prediction.eq(target).float().mean().item()),
        "train/masked_token_ratio": float(noisy_target.eq(MASK_TOKEN).float().mean().item()),
        "train/logit_entropy": float(
            -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1).mean().item()
        ),
        "train/logit_std": float(logits.detach().float().std(unbiased=False).item()),
        "train/top1_confidence": float(confidence.mean().item()),
        "train/top1_confidence_std": float(confidence.std(unbiased=False).item()),
        "train/prediction_unique_tokens": float(unique_counts.mean().item()),
        "train/prediction_dominant_token_share": float(dominant_token_shares.mean().item()),
        "train/active_supports": float(support_mask.float().sum(dim=1).mean().item()),
    }

    histogram_tensors = {
        "confidence": confidence.detach().cpu(),
    }
    if attention_weights is None:
        return scalar_metrics, histogram_tensors

    attention_weights = attention_weights.detach().float()
    attention_mask = support_mask.to(dtype=torch.bool)
    valid_counts = attention_mask.sum(dim=1)
    attention_entropy = -(attention_weights.clamp_min(1e-8).log() * attention_weights).sum(dim=1)
    normalized_attention_entropy = torch.where(
        valid_counts > 1,
        attention_entropy / valid_counts.float().log(),
        torch.zeros_like(attention_entropy),
    )
    masked_attention = attention_weights.masked_fill(~attention_mask, 0.0)
    scalar_metrics["train/support_attention_max"] = float(masked_attention.max(dim=1).values.mean().item())
    scalar_metrics["train/support_attention_entropy"] = float(normalized_attention_entropy.mean().item())
    histogram_tensors["support_attention"] = attention_weights[attention_mask].detach().cpu()
    return scalar_metrics, histogram_tensors


def _log_training_scalars(writer, step: int, loss: float, lr: float, scalar_metrics: dict[str, float]) -> None:
    writer.add_scalar("train/loss", loss, step)
    writer.add_scalar("train/learning_rate", lr, step)
    for metric_name, metric_value in scalar_metrics.items():
        writer.add_scalar(metric_name, metric_value, step)


def _log_model_histograms(writer, model: UNet, histogram_tensors: dict[str, torch.Tensor], step: int) -> None:
    writer.add_histogram("model/output_head_weights", model.out[-1].weight.detach().float().cpu(), step)
    writer.add_histogram("model/token_embedding_weights", model.token_embedding.weight.detach().float().cpu(), step)
    for histogram_name, histogram_values in histogram_tensors.items():
        if histogram_values.numel() == 0:
            continue
        writer.add_histogram(f"train/{histogram_name}_distribution", histogram_values, step)


def _log_validation_summary(writer, preview_dir: Path, summary: dict[str, object], step: int) -> None:
    overall_summary = summary.get("overall", {})
    if isinstance(overall_summary, dict):
        for metric_name in ("mean_pixel_accuracy", "mean_rgb_mae", "exact_match_count"):
            metric_value = overall_summary.get(metric_name)
            if metric_value is not None:
                writer.add_scalar(f"validation/overall/{metric_name}", metric_value, step)

    packs = summary.get("packs", {})
    if not isinstance(packs, dict):
        return

    for pack_name, pack_summary in packs.items():
        if not isinstance(pack_summary, dict):
            continue

        pack_tag = _sanitize_tag_component(str(pack_name))
        summary_path = preview_dir / str(pack_name) / "summary.png"
        if summary_path.exists():
            with Image.open(summary_path) as summary_image:
                writer.add_image(
                    f"validation/packs/{pack_tag}/summary",
                    np.asarray(summary_image),
                    step,
                    dataformats="HWC",
                )

        for metric_name in ("mean_pixel_accuracy", "mean_rgb_mae", "exact_match_count"):
            metric_value = pack_summary.get(metric_name)
            if metric_value is not None:
                writer.add_scalar(f"validation/packs/{pack_tag}/{metric_name}", metric_value, step)


@torch.no_grad()
def _write_validation_preview(
    model: UNet,
    dataset: TextureDataset,
    checkpoint_dir: Path,
    step: int,
) -> tuple[Path, dict[str, object]] | None:
    if len(dataset) == 0:
        return None

    was_training = model.training
    model.eval()
    preview_dir = _validation_preview_dir(checkpoint_dir, step)
    summary = write_validation_matrix(
        model=model,
        dataset=dataset,
        output_dir=preview_dir,
        checkpoint_path=checkpoint_dir / "latest.pt",
    )
    if was_training:
        model.train()
    return preview_dir, summary


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

    train_dataset = TextureDataset(
        split="train",
        min_support_exemplars=MIN_SUPPORT_EXEMPLARS,
        max_support_exemplars=MAX_SUPPORT_EXEMPLARS,
    )
    val_dataset = TextureDataset(
        split="val",
        min_support_exemplars=MIN_SUPPORT_EXEMPLARS,
        max_support_exemplars=MAX_SUPPORT_EXEMPLARS,
    )
    if len(train_dataset) == 0:
        raise ValueError("Training split is empty. Run preprocessing first.")

    batch_size = 2 if device.type == "cuda" else 1
    grad_accum_steps = 16 if device.type == "cuda" else 1
    validation_interval = 2_000
    save_interval = 500
    history_flush_interval = 10

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_training_batch,
    )
    train_iter = iter(train_loader)

    model = UNet(vocab_size=VOCAB_SIZE).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=1e-6)
    checkpoint_path = _latest_checkpoint_path(checkpoint_dir)
    start_step = _maybe_load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, steps)
    metrics_path = _metrics_history_path(checkpoint_dir)
    metric_history = _load_metric_history(metrics_path, max_step=start_step)
    pending_metric_history: list[MetricRecord] = []
    if start_step >= steps:
        _write_training_graphs(checkpoint_dir, metric_history)
        print(f"Checkpoint already at step {start_step}, target was {steps}; nothing to do.")
        return checkpoint_path

    tensorboard_dir = _tensorboard_log_dir(checkpoint_dir)
    writer = _make_summary_writer(tensorboard_dir, start_step)
    print(f"TensorBoard logs: {tensorboard_dir.resolve()}")

    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    scaler = _make_grad_scaler(device)
    tensorboard_histogram_interval = validation_interval

    try:
        model.train()
        for step in range(start_step, steps):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            batch_scalar_metrics: dict[str, float] = {}
            histogram_tensors: dict[str, torch.Tensor] = {}

            for accum_idx in range(grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                content_ref = batch["content_ref"].to(device)
                support_content_refs = batch["support_content_refs"].to(device)
                support_style_refs = batch["support_style_refs"].to(device)
                support_mask = batch["support_mask"].to(device)
                target = batch["target"].to(device)

                content_ref, support_content_refs, support_style_refs, support_mask = _apply_cfg_dropout(
                    content_ref,
                    support_content_refs,
                    support_style_refs,
                    support_mask,
                )
                t = torch.randint(1, NUM_TIMESTEPS + 1, (target.shape[0],), device=device)
                noisy_target = apply_mask(target, t)
                capture_diagnostics = accum_idx == grad_accum_steps - 1

                with autocast_context():
                    if capture_diagnostics:
                        logits, aux = model(
                            noisy_target,
                            content_ref,
                            support_content_refs,
                            support_style_refs,
                            support_mask,
                            t,
                            return_aux=True,
                        )
                    else:
                        logits = model(
                            noisy_target,
                            content_ref,
                            support_content_refs,
                            support_style_refs,
                            support_mask,
                            t,
                        )
                        aux = None
                    loss = F.cross_entropy(logits, target) / grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += float(loss.detach().item()) * grad_accum_steps

                if capture_diagnostics:
                    batch_scalar_metrics, histogram_tensors = _collect_batch_diagnostics(
                        logits=logits,
                        target=target,
                        noisy_target=noisy_target,
                        support_mask=support_mask,
                        attention_weights=None if aux is None else aux["support_attention_weights"],
                    )

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            batch_scalar_metrics["model/grad_norm"] = _gradient_norm(model)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            batch_scalar_metrics["model/parameter_norm"] = _parameter_norm(model)
            batch_scalar_metrics["model/output_head_weight_std"] = float(
                model.out[-1].weight.detach().float().std(unbiased=False).item()
            )
            batch_scalar_metrics["model/token_embedding_weight_std"] = float(
                model.token_embedding.weight.detach().float().std(unbiased=False).item()
            )

            current_step = step + 1
            average_loss = total_loss / grad_accum_steps
            lr = scheduler.get_last_lr()[0]
            metric_record: MetricRecord = {
                "step": current_step,
                "loss": average_loss,
                "lr": lr,
            }
            metric_history.append(metric_record)
            pending_metric_history.append(metric_record)
            _log_training_scalars(writer, current_step, average_loss, lr, batch_scalar_metrics)

            if current_step == 1 or current_step % history_flush_interval == 0 or current_step == steps:
                _append_metric_history(metrics_path, pending_metric_history)
                pending_metric_history.clear()
                writer.flush()

            if current_step == 1 or current_step % 10 == 0 or current_step == steps:
                print(f"step={current_step}/{steps} loss={average_loss:.4f} lr={lr:.6e}")

            if current_step == 1 or current_step % tensorboard_histogram_interval == 0 or current_step == steps:
                _log_model_histograms(writer, model, histogram_tensors, current_step)

            if current_step % save_interval == 0 or current_step == steps:
                if pending_metric_history:
                    _append_metric_history(metrics_path, pending_metric_history)
                    pending_metric_history.clear()
                _save_checkpoint(checkpoint_path, model, optimizer, scheduler, current_step, steps)
                step_checkpoint = checkpoint_dir / f"step_{current_step:06d}.pt"
                _save_checkpoint(step_checkpoint, model, optimizer, scheduler, current_step, steps)
                _write_training_graphs(checkpoint_dir, metric_history)

            if current_step % validation_interval == 0 or current_step == steps:
                preview_result = _write_validation_preview(model, val_dataset, checkpoint_dir, current_step)
                if preview_result is not None:
                    preview_dir, summary = preview_result
                    _log_validation_summary(writer, preview_dir, summary, current_step)
                    writer.flush()

        return checkpoint_path
    finally:
        writer.close()
