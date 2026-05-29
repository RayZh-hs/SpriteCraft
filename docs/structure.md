# Project Structure

`SpriteCraft` is currently organized around a vanilla-anchored, per-pack training pipeline with a dual-model architecture.

## Top Level

- `src/spritecraft/`: main package.
- `data/`: raw pack archives, optional `manifest.json`, and generated processed datasets.
- `checkpoints/`: training runs and per-pack checkpoints.
- `output/`: generated texture bundles and evaluation artifacts.
- `docs/`: legacy planning docs only; now intentionally reduced.

## Package Layout

- `src/spritecraft/cli.py`: CLI entry point. Exposes `preprocess`, `train`, `train-recolor`, `generate`, and `debug`.
- `src/spritecraft/config.py`: shared paths, image size, timestep count, support-ref limits, validation filenames.
- `src/spritecraft/data/`
  - `preprocess.py`: extracts pack archives, filters full-cube block textures, resizes images, and writes per-pack datasets.
  - `dataset.py`: loads one pack dataset at a time and samples style references from that pack.
  - `support_index.py`: heuristic texture-descriptor and support-ranking helpers.
- `src/spritecraft/models/`
  - `unet.py`: `StyleAwareUNet` (~7.7M params), residual blocks, cross-attention, timestep embedding.
  - `recolor.py`: `RecolorNet` (~1.0M params), lightweight U-Net for structure-preserving color transfer; `RecolorLoss` for training.
  - `diffusion.py`: cosine beta schedule, noise/denoising helpers, DDPM sampling step.
- `src/spritecraft/training/`
  - `train.py`: per-pack diffusion training loop, checkpointing, validation previews, TensorBoard logging.
  - `recolor_train.py`: per-pack RecolorNet training loop.
  - `content_loss.py`: content-aware losses (structure, contrast, hue, color moment) with gated spatial weighting.
- `src/spritecraft/inference/`
  - `generate.py`: load dual-model checkpoints and generate bundles for selected textures with tiered candidate selection.
  - `sampler.py`: iterative denoising sampler with dual-model fallback, detail injection, metrics, bundle saving.
  - `evaluate.py`: validation-matrix rendering and summary export.
- `src/spritecraft/debug/`: runtime status, snapshots, preview requests for live training inspection.

## Data Outputs

- `data/processed/pack_report.json`: preprocessing summary across all discovered packs.
- `data/processed/packs/<pack_id>/dataset.npz`: per-pack RGB arrays for train/val content and targets, plus all target textures for style-reference sampling.
- `data/processed/packs/<pack_id>/pair_index.json`: filenames, base pack id, and style metadata for that pack.

## Training / Generation Outputs

- `checkpoints/<run>/<pack_id>/latest.pt`: latest StyleAwareUNet checkpoint for one target pack.
- `checkpoints/<run>/<pack_id>/recolor_latest.pt`: latest RecolorNet checkpoint for one target pack.
- `checkpoints/<run>/<pack_id>/step_*.pt`: step snapshots.
- `checkpoints/<run>/<pack_id>/training_metrics.csv`: scalar loss / LR history.
- `checkpoints/<run>/<pack_id>/validation/step_*/`: side-by-side validation previews and debug diagnostic panels.
- `checkpoints/<run>/tensorboard/<pack_id>/`: TensorBoard logs grouped by pack.
- `checkpoints/<run>/<pack_id>/.spritecraft-runtime/`: debug heartbeat and request files.
- `output/<pack_id>/<bundle>/`: generated texture, comparison image, metadata, and metrics for one sample.
