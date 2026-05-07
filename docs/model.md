# Model Status

## Current Model

- Training is **per target pack**. Running `spritecraft train` without `--pack` trains one checkpoint per available processed pack.
- The dataset is **vanilla-anchored**: each sample pairs one base-pack texture with the matching texture from one target pack.
- The pipeline is **continuous RGB**, not palette-token based. Preprocessing stores float RGB and alpha arrays in `dataset.npz`.
- Conditioning uses:
  - `content_rgb`: the base / vanilla texture.
  - `style_refs`: `1..3` textures from the same target pack.
  - `t`: diffusion timestep.
- The network is `StyleAwareUNet` in `src/spritecraft/models/unet.py`: content encoder, style encoder, cross-attention at `8x8`, timestep MLP, and a small decoder back to RGB.
- Diffusion utilities use a **20-step Gaussian schedule** in `src/spritecraft/models/diffusion.py`.
- Training loss is `L1 + 0.1 *` a simplified SSIM-like term.

## What Is Implemented

- Preprocessing filters to full-cube block textures by reading blockstates and model face textures from the base pack.
- Images are resized to `32x32`; `16x` uses nearest-neighbor and `64x` uses Lanczos.
- Validation is filename-based using the fixed set in `config.py`.
- Inference starts from noise and iteratively denoises, then writes bundle directories with images, metrics, and metadata.
- Runtime debug support exists for status, snapshots, and preview generation during training.

## Current Gaps / Leftovers

- `StyleAwareUNet.forward()` receives `noisy_target`, but does **not** actually encode or use it. In practice, the current model is conditioned by content, style refs, and timestep, while the outer training / sampling loop is still shaped like diffusion.
- Alpha channels are preprocessed and loaded, but the model and loss currently operate on RGB only.
- The dataset and model support up to 3 style references, but `generate.py` currently uses `MIN_SUPPORT_EXEMPLARS`, so generation is effectively **1-shot** by default.
- `support_index.py` exists, but current training / generation sample support textures directly rather than using the ranking helpers.
- `inference/export.py` and `config.PALETTE_PATH` are leftovers from the older discrete / palette pipeline and are not part of the current RGB training path.
- `inference/evaluate.py` exists for validation matrices, but it is not exposed through the CLI today.
