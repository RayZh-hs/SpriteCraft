"""Shared runtime metadata helpers for training/debug workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

RUNTIME_DIRNAME = ".spritecraft-runtime"
STATUS_PREFIX = "trainer_"
REQUESTS_PREFIX = "requests_"
DEBUG_DIRNAME = "debug"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temp_path.replace(path)


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def runtime_dir(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / RUNTIME_DIRNAME


def runtime_status_path(checkpoint_dir: str | Path, pid: int) -> Path:
    return runtime_dir(checkpoint_dir) / f"{STATUS_PREFIX}{pid}.json"


def runtime_request_dir(checkpoint_dir: str | Path, pid: int) -> Path:
    return runtime_dir(checkpoint_dir) / f"{REQUESTS_PREFIX}{pid}"


def list_request_paths(checkpoint_dir: str | Path, pid: int) -> list[Path]:
    request_root = runtime_request_dir(checkpoint_dir, pid)
    if not request_root.exists():
        return []
    return sorted(request_root.glob("*.json"))


def snapshots_dir(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / DEBUG_DIRNAME / "snapshots"


def previews_dir(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / DEBUG_DIRNAME / "previews"


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
