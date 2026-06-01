# Model Status

## Current Dual-Model Architecture

The pipeline uses two complementary models working together:

### StyleAwareUNet (~7.7M parameters)
Primary diffusion model for texture generation. Architecture:
- Early fusion of noisy target + vanilla content at the input.
- Encoder: 32→16→8 spatial compression with `ResBlock` + strided conv.
- Multi-reference style encoder: 3 reference textures → 8×8 style feature maps.
- Masked cross-attention at the bottleneck (8×8), with content queries attending to style keys/values.
- Sinusoidal timestep MLP injected at the bottleneck.
- Decoder with spatial skip connections from the encoder at 32×32 and 16×16.
- Predicts **diffusion noise (epsilon)**, not clean RGB directly.

### RecolorNet (~1.0M parameters)
Lightweight U-Net for structure-preserving color transfer. Serves as a fallback for complex/zero-shot textures where diffusion produces structurally incoherent outputs. Architecture:
- Content encoder + style encoder (3 reference textures, averaged with mask).
- Style conditioning injected at the bottleneck.
- Decoder with skip connection to raw content RGB (structure preservation).
- Predicts a residual added to the content → final output in [0, 1].

## Training

- Training is **per target pack**. Two separate training commands:
  - `spritecraft train`: trains StyleAwareUNet.
  - `spritecraft train-recolor`: trains RecolorNet.
- Dataset is **vanilla-anchored**: each sample pairs one base-pack texture with the matching texture from one target pack.
- The pipeline is **continuous RGB**, not palette-token based. Preprocessing stores float RGB arrays in `dataset.npz`.
- Alpha channels are preprocessed and loaded but the model and loss operate on RGB only.

### StyleAwareUNet Training
- **Diffusion schedule**: 40-step cosine beta schedule (`s=0.008, max_beta=0.999`), ensures near-zero terminal SNR.
- **Noise prediction**: model predicts epsilon; x₀ recovered via `predict_x0_from_noise`.
- **Gradient clipping**: `max_norm=1.0` applied.
- **Mixed precision**: bfloat16 autocast on supported CUDA devices.
- Optimizer: AdamW (`lr=3e-4, weight_decay=0.01`), CosineAnnealingLR to `eta_min=1e-6`.

### RecolorNet Training
- **Loss**: reconstruction L1 + edge preservation + color moment matching + content identity regularization.
- **Weights**: recon=1.0, edge=0.5, color=1.5, content_identity=0.1.

## Loss Functions (StyleAwareUNet)

```
total = noise_MSE + 0.70×recon_L1 + 0.30×gradient + 0.30×content_loss
```

### Gradient Loss
- Luminance gradient L1: forward-difference edge preservation on Y channel.
- Per-channel RGB gradient L1 (weight 0.5): penalizes color bleeding and channel-specific blurring.

### Content Loss (weight 0.30)
Spatially gated, content-relative losses:

| Component | Weight | Description |
|-----------|--------|-------------|
| structure | 1.0 | Gradient delta L1 + 0.5×detail delta L1 + 0.75×Laplacian delta L1 (source-relative) |
| contrast | 2.5 | Gated under-contrast penalty (squared gate, one-sided) + 0.6×edge shortfall |
| hue | 3.0 | Opponent-chroma angular alignment, gated by target chroma (γ=0.7), plus 0.25×chroma strength |
| color moment | 1.5 | Per-channel mean/std deviation matching (non-spatial, fast convergence) |

Key design decisions:
- **Source-relative**: structure/comparison losses operate on *deltas from the vanilla content*, not raw targets. The model learns to predict only what *changes*.
- **Target-gated**: contrast and hue losses are gated by the target's local contrast/chroma so flat regions are ignored. The gate uses a squared power (γ=2.0) to focus penalty on high-detail regions.
- **One-sided contrast**: only penalizes under-contrast (F.relu(target - pred)), not oversharpening. Over-sharpening is already constrained by reconstruction and structure losses.
- **Detail emphasis**: raises the penalty for textures whose target contains concentrated detail (top-12.5% local std values).

## Conditioning

- `noisy_target`: current diffusion sample [B, 3, 32, 32].
- `content_rgb`: base/vanilla texture [B, 3, 32, 32].
- `style_refs`: 3 ranked textures from the same target pack [B, 3, 3, 32, 32].
- `style_ref_mask`: boolean validity mask for padded references [B, 3].
- `t`: diffusion timestep [B].

## Inference

### Sampling
- 40 DDPM reverse steps with x₀ clipping [0, 1].
- Starts from pure Gaussian noise.

### Dual-Model Candidate Selection (tiered)
1. **Diffusion first**: sample 2 stochastic candidates. If the best diffusion score ≥ 0.50 (support descriptor cosine similarity minus sharpness penalty), use it.
2. **RecolorNet fallback**: if available and score ≥ 0.40 within 0.25 margin, use the learned recolor.
3. **Deterministic recolor**: support-pair affine color migration or style-stat recolor as last resort.
4. **Detail injection**: for the best diffusion candidate, try support-pair detail fusion at amounts [0.35, 0.65]. The fusion starts from the original diffusion prediction and only injects missing high-frequency residuals from the deterministic recolor where local layouts still agree, avoiding the global blur caused by rebuilding the image from a blurred diffusion base.
5. **Source-relative style check**: candidate style is scored not only against support descriptors, but also by how closely its change-from-vanilla matches the support pack's typical change magnitude. This penalizes recolor outputs that stay too close to vanilla on stylized packs.

## What Is Implemented

- Preprocessing filters to full-cube block textures by reading blockstates and model face textures from the base pack.
- Images are resized to `32×32`; `16×` uses nearest-neighbor and `64×` uses Lanczos.
- Validation is filename-based using the fixed set in `config.py`.
- Training samples ranked support textures using vanilla/content descriptors and a filename-family prior from `support_index.py`.
- Runtime debug support for status, snapshots, and preview generation during training.
- `inference/evaluate.py` exists for validation matrices but is not exposed through the CLI.

## Current Gaps / Leftovers

- Alpha channels are preprocessed and loaded, but the model and loss operate on RGB only.
- `64× → 32×` preprocessing uses Lanczos globally; realistic packs may benefit from a pack-specific or configurable resize policy.
- Support ranking is heuristic rather than learned.
- `inference/export.py` and `config.PALETTE_PATH` are leftovers from the older discrete/palette pipeline and are not part of the current RGB training path.
- Glass and transparent/line-art textures remain difficult — need dedicated handling.
- Complex tiling textures (oak_leaves, tnt_side) need more structure-aware training.
