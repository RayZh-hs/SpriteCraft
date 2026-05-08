"""Runtime debug commands for live and offline training inspection."""

from __future__ import annotations

from datetime import datetime, timezone
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from spritecraft.config import CHECKPOINTS_DIR
from spritecraft.debug.utility import (
    STATUS_PREFIX,
    is_pid_alive,
    load_json,
    previews_dir,
    runtime_request_dir,
    snapshots_dir,
    utcnow_iso,
    write_json_atomic,
)
from spritecraft.inference import evaluate
from spritecraft.inference.sampler import _latest_checkpoint_path


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: object, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    return default


def _discover_running_instances(checkpoints_root: str | Path = CHECKPOINTS_DIR) -> list[dict[str, object]]:
    root = Path(checkpoints_root)
    status_paths = sorted(root.glob(f"**/.spritecraft-runtime/{STATUS_PREFIX}*.json"))
    instances: list[dict[str, object]] = []
    for status_path in status_paths:
        try:
            status = load_json(status_path)
        except (OSError, ValueError):
            continue
        pid = _coerce_int(status.get("pid", -1), -1)
        if pid <= 0 or not is_pid_alive(pid):
            continue
        status["pid"] = pid
        checkpoint_dir_raw = _coerce_str(status.get("checkpoint_dir"))
        if not checkpoint_dir_raw:
            continue
        status["checkpoint_dir"] = str(Path(checkpoint_dir_raw).resolve())
        status["status_path"] = str(status_path.resolve())
        instances.append(status)
    return sorted(instances, key=lambda item: _coerce_int(item.get("pid", 0)))


def _known_issues(status: dict[str, object]) -> list[str]:
    issues: list[str] = []
    step = _coerce_int(status.get("step", 0))
    last_saved_step = _coerce_int(status.get("last_saved_step", 0))
    last_validation_step = _coerce_int(status.get("last_validation_step", 0))
    updated_at_raw = status.get("updated_at")
    last_preview_dir = status.get("last_preview_dir")

    if step > last_saved_step:
        issues.append(f"Latest checkpoint lags live model by {step - last_saved_step} step(s).")
    if step > last_validation_step:
        issues.append(f"Preview images lag live model by {step - last_validation_step} step(s).")
    if not last_preview_dir:
        issues.append("No preview images have been generated yet.")
    if isinstance(updated_at_raw, str):
        try:
            updated_at = datetime.fromisoformat(updated_at_raw)
        except ValueError:
            updated_at = None
        if updated_at is not None:
            age_s = (datetime.now(timezone.utc) - updated_at).total_seconds()
            if age_s > 60:
                issues.append(f"Runtime heartbeat is stale ({age_s:.0f}s old).")
    return issues


def _print_instance_table(instances: list[dict[str, object]]) -> None:
    print("Running SpriteCraft instances:")
    for status in instances:
        checkpoint_dir = Path(str(status["checkpoint_dir"]))
        print(
            f"  pid={status['pid']} "
            f"step={status.get('step', 0)}/{status.get('target_steps', '?')} "
            f"device={status.get('device', '?')} "
            f"checkpoint_dir={checkpoint_dir}"
        )


def _select_instance(instances: list[dict[str, object]], pid: int | None) -> dict[str, object] | None:
    if not instances:
        return None
    if pid is not None:
        for instance in instances:
            if _coerce_int(instance.get("pid", 0)) == pid:
                return instance
        raise ValueError(f"No running trainer with pid={pid} was found.")
    if len(instances) == 1:
        return instances[0]

    _print_instance_table(instances)
    valid_pids = {_coerce_int(instance.get("pid", 0)) for instance in instances}
    while True:
        chosen = input("Multiple trainers are running. Enter pid to attach: ").strip()
        try:
            chosen_pid = int(chosen)
        except ValueError:
            print("Please enter a numeric pid.")
            continue
        if chosen_pid in valid_pids:
            for instance in instances:
                if _coerce_int(instance.get("pid", 0)) == chosen_pid:
                    return instance
        print(f"pid={chosen_pid} is not in the running instance list.")


def _request_path(instance: dict[str, object], action: str) -> Path:
    checkpoint_dir = Path(str(instance["checkpoint_dir"]))
    pid = _coerce_int(instance.get("pid", 0))
    request_id = f"{action}_{int(time.time() * 1000)}"
    request_path = runtime_request_dir(checkpoint_dir, pid) / f"{request_id}.json"
    write_json_atomic(
        request_path,
        {
            "id": request_id,
            "action": action,
            "pid": pid,
            "checkpoint_dir": str(checkpoint_dir),
            "status": "pending",
            "created_at": utcnow_iso(),
        },
    )
    return request_path


def _wait_for_request(request_path: Path, timeout_s: float) -> dict[str, object]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if request_path.exists():
            payload = load_json(request_path)
            status = payload.get("status")
            if status in {"completed", "failed"}:
                return payload
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {request_path.name} to complete.")


def _resolve_checkpoint_path(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_file():
        return checkpoint_dir
    return _latest_checkpoint_path(checkpoint_dir)


def _offline_snapshot(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_dir)
    snapshot_root = snapshots_dir(checkpoint_dir)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = snapshot_root / f"offline_snapshot_{timestamp}{checkpoint_path.suffix}"
    shutil.copy2(checkpoint_path, snapshot_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    report_path = snapshot_root / f"offline_snapshot_{timestamp}.json"
    write_json_atomic(
        report_path,
        {
            "mode": "offline",
            "source_checkpoint": str(checkpoint_path.resolve()),
            "snapshot_path": str(snapshot_path.resolve()),
            "step": int(checkpoint.get("step", 0)),
            "target_steps": int(checkpoint.get("target_steps", 0)),
            "created_at": utcnow_iso(),
            "known_issues": [
                "Snapshot reflects the latest saved checkpoint, not any unsaved in-memory training progress."
            ],
        },
    )
    return snapshot_path


def _offline_preview(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = previews_dir(checkpoint_dir) / f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    evaluate.run(checkpoint_dir=checkpoint_dir, output_dir=output_dir, split="val", mode="matrix")
    return output_dir


def _print_status(instance: dict[str, object]) -> None:
    issues = _known_issues(instance)
    print(f"pid: {instance['pid']}")
    print(f"status: {instance.get('status', 'running')}")
    print(f"checkpoint_dir: {instance['checkpoint_dir']}")
    print(f"device: {instance.get('device', '?')}")
    print(f"step: {instance.get('step', 0)}/{instance.get('target_steps', '?')}")
    print(f"loss: {instance.get('last_loss')}")
    print(f"learning_rate: {instance.get('last_lr')}")
    print(f"last_saved_step: {instance.get('last_saved_step', 0)}")
    print(f"last_validation_step: {instance.get('last_validation_step', 0)}")
    print(f"last_preview_dir: {instance.get('last_preview_dir') or '(none)'}")
    if issues:
        print("known_issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("known_issues: none")


def run(
    action: str = "status",
    pid: int | None = None,
    checkpoint_dir: str | Path = CHECKPOINTS_DIR,
    wait_timeout: float = 600.0,
) -> None:
    instances = _discover_running_instances()
    instance = _select_instance(instances, pid)

    if action == "status":
        if instance is None:
            print("No running SpriteCraft training instances were found.")
            return
        _print_status(instance)
        return

    if instance is not None:
        request_path = _request_path(instance, action)
        response = _wait_for_request(request_path, timeout_s=wait_timeout)
        if response.get("status") == "failed":
            raise RuntimeError(str(response.get("error", f"{action} request failed.")))
        result: dict[str, Any]
        result_raw = response.get("result", {})
        if isinstance(result_raw, dict):
            result = result_raw
        else:
            result = {}
        if action == "snapshot":
            print(f"Saved live snapshot to {result.get('snapshot_path')}")
            print(f"Saved snapshot report to {result.get('report_path')}")
        elif action == "preview":
            print(f"Saved live preview bundle to {result.get('preview_dir')}")
            print("TensorBoard images were updated under validation/packs/<pack>/summary.")
        return

    checkpoint_dir = Path(checkpoint_dir)
    if action == "snapshot":
        snapshot_path = _offline_snapshot(checkpoint_dir)
        print(f"Saved offline snapshot to {snapshot_path}")
        return
    if action == "preview":
        preview_dir = _offline_preview(checkpoint_dir)
        print(f"Saved offline preview bundle to {preview_dir}")
        return
    raise ValueError(f"Unsupported debug action {action!r}")
