"""Iterative decoding sampler."""

from pathlib import Path

from spritecraft.config import OUTPUT_DIR


def run(content_path: str, style_path: str, output_dir: str = OUTPUT_DIR):
    """Generate a texture from content and style references."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Implement sampling loop
    raise NotImplementedError("Sampler not yet implemented")
