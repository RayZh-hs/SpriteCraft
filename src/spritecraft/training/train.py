"""Training loop and utilities."""

from pathlib import Path

from spritecraft.config import CHECKPOINTS_DIR


def run(checkpoint_dir: str | Path = CHECKPOINTS_DIR, steps: int = 100_000):
    """Run training loop."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Implement training loop
    raise NotImplementedError("Training loop not yet implemented")
