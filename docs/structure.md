# Project Structure

`SpriteCraft` is currently organized around a vanilla-anchored, per-pack training pipeline.

## Top Level

- `src/spritecraft/`: main package.
- `data/`: raw pack archives, optional `manifest.json`, and generated processed datasets.
- `checkpoints/`: training runs and per-pack checkpoints.
- `output/`: generated texture bundles and evaluation artifacts.
- `docs/`: legacy planning docs only; now intentionally reduced.

## Package Layout

- `src/spritecraft/cli.py`: CLI entry point. Exposes `preprocess`, `train`, `generate`, and `debug`.
- `src/spritecraft/config.py`: shared paths, image size, timestep count, support-ref limits, validation filenames.
- `src/spritecraft/data/`
  - `preprocess.py`: extracts pack archives, filters full-cube block textures, resizes images, and writes per-pack datasets.
  - `dataset.py`: loads one pack dataset at a time and samples style references from that pack.
  - `support_index.py`: heuristic texture-descriptor and support-ranking helpers.
- `src/spritecraft/models/`
  - `unet.py`: `StyleAwareUNet`, residual blocks, cross-attention, timestep embedding.
  - `diffusion.py`: Gaussian noise schedule and sampling helpers.
- `src/spritecraft/training/train.py`: per-pack training loop, checkpointing, validation previews, TensorBoard logging.
- `src/spritecraft/inference/`
  - `generate.py`: load a pack checkpoint and generate bundles for selected textures.
  - `sampler.py`: iterative denoising sampler, metrics, bundle saving.
  - `evaluate.py`: validation-matrix rendering and summary export.
- `src/spritecraft/debug/`: runtime status, snapshots, preview requests for live training inspection.

## Data Outputs

- `data/processed/pack_report.json`: preprocessing summary across all discovered packs.
- `data/processed/packs/<pack_id>/dataset.npz`: per-pack RGB arrays for train/val content and targets, plus all target textures for style-reference sampling.
- `data/processed/packs/<pack_id>/pair_index.json`: filenames, base pack id, and style metadata for that pack.

## Training / Generation Outputs

- `checkpoints/<run>/<pack_id>/latest.pt`: latest checkpoint for one target pack.
- `checkpoints/<run>/<pack_id>/step_*.pt`: step snapshots.
- `checkpoints/<run>/<pack_id>/training_metrics.csv`: scalar loss / LR history.
- `checkpoints/<run>/<pack_id>/validation/step_*/`: side-by-side validation previews.
- `checkpoints/<run>/tensorboard/<pack_id>/`: TensorBoard logs grouped by pack.
- `checkpoints/<run>/<pack_id>/.spritecraft-runtime/`: debug heartbeat and request files.
- `output/<pack_id>/<bundle>/`: generated texture, comparison image, metadata, and metrics for one sample.
