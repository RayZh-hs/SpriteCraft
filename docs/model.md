# Model Status

## Current Model

- Training is **per target pack**. Running `spritecraft train` without `--pack` trains one checkpoint per available processed pack.
- The dataset is **vanilla-anchored**: each sample pairs one base-pack texture with the matching texture from one target pack.
- The pipeline is **continuous RGB**, not palette-token based. Preprocessing stores float RGB and alpha arrays in `dataset.npz`.
- Conditioning uses:
  - `noisy_target`: the current diffusion sample.
  - `content_rgb`: the base / vanilla texture.
  - `style_refs`: up to `3` ranked textures from the same target pack.
  - `t`: diffusion timestep.
- The network is `StyleAwareUNet` in `src/spritecraft/models/unet.py`: noisy/content fusion at input, multi-reference style encoder, masked cross-attention at `8x8`, timestep MLP, and spatial skip connections in the decoder.
- Diffusion utilities use a **20-step Gaussian schedule** in `src/spritecraft/models/diffusion.py`.
- The model predicts **diffusion noise (`epsilon`)**, not clean RGB directly.
- Training loss is `MSE(noise) + 0.5 * L1(x0) + 0.25 * gradient(x0)` to preserve local edges.

## What Is Implemented

- Preprocessing filters to full-cube block textures by reading blockstates and model face textures from the base pack.
- Images are resized to `32x32`; `16x` uses nearest-neighbor and `64x` uses Lanczos.
- Validation is filename-based using the fixed set in `config.py`.
- Training samples ranked support textures using vanilla/content descriptors and a filename-family prior from `support_index.py`.
- Inference starts from noise, uses the same epsilon-prediction formulation as training, and writes bundle directories with images, metrics, and metadata.
- Runtime debug support exists for status, snapshots, and preview generation during training.

## Current Gaps / Leftovers

- Alpha channels are preprocessed and loaded, but the model and loss currently operate on RGB only.
- `64x -> 32x` preprocessing still uses Lanczos globally; realistic packs may benefit from a pack-specific or configurable resize policy.
- Training and generation now default to **3-shot** support when available, but support ranking is still heuristic rather than learned.
- `inference/export.py` and `config.PALETTE_PATH` are leftovers from the older discrete / palette pipeline and are not part of the current RGB training path.
- `inference/evaluate.py` exists for validation matrices, but it is not exposed through the CLI today.
