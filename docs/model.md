# Model Status

## Current Dual-Model Architecture

The pipeline uses two complementary models working together:

### StyleAwareUNet (~7.7M parameters)

Primary diffusion model for texture generation. Architecture:

- Early fusion of noisy target + vanilla content at the input.
- Encoder: 32x32 -> 16x16 -> 8x8 spatial compression with `ResBlock` + strided conv.
- Multi-reference style encoder: 3 reference textures -> 8x8 style feature maps.
- Masked cross-attention at the bottleneck, with content queries attending to style keys/values.
- Sinusoidal timestep MLP injected at the bottleneck.
- Decoder with spatial skip connections from the encoder at 32x32 and 16x16.
- Predicts diffusion noise (`epsilon`), not clean RGB directly.

### RecolorNet (~1.0M parameters)

Lightweight U-Net for structure-preserving color transfer. It serves as a
fallback for complex or zero-shot textures where diffusion produces
structurally incoherent outputs. Architecture:

- Content encoder + style encoder with 3 reference textures, averaged with a validity mask.
- Style conditioning injected at the bottleneck.
- Decoder with skip connection to raw content RGB for structure preservation.
- Predicts a residual added to the content, clamped to `[0, 1]`.

## Training

- Training is per target pack.
- `spritecraft train --mode recolor_first` trains RecolorNet and then StyleAwareUNet for each pack.
- `spritecraft train --mode recolor_only` trains only RecolorNet.
- `spritecraft train --mode std_only` trains only StyleAwareUNet.
- Dataset samples are vanilla-anchored: each sample pairs one base-pack texture with the matching texture from one target pack.
- The pipeline is continuous RGB, not palette-token based. Preprocessing stores float RGB arrays in `dataset.npz`.
- Alpha channels are preprocessed and loaded, but the model and loss operate on RGB only.

### StyleAwareUNet Training

- Diffusion schedule: 40-step cosine beta schedule (`s=0.008`, `max_beta=0.999`), ensuring near-zero terminal SNR.
- Noise prediction: the model predicts `epsilon`; `x0` is recovered with `predict_x0_from_noise`.
- Gradient clipping: `max_norm=1.0`.
- Mixed precision: bfloat16 autocast on supported CUDA devices.
- Optimizer: AdamW (`lr=3e-4`, `weight_decay=0.01`).
- LR schedule: 5% warmup followed by cosine annealing to `eta_min=1e-6`.
- EMA decay: `0.995`.
- Resume behavior restores optimizer and scheduler state when compatible, including when training is extended to a larger target step count.

### RecolorNet Training

- Loss: reconstruction L1 + edge preservation + color moment matching + content identity regularization.
- Weights: recon=1.0, edge=0.5, color=1.5, content_identity=0.1.
- Optimizer: AdamW (`lr=3e-4`, `weight_decay=0.03`).
- LR schedule: 5% warmup followed by cosine annealing to `eta_min=1e-6`.

## Loss Functions (StyleAwareUNet)

The diffusion trainer uses two objectives. The route is selected per training
sample by comparing the target texture against the vanilla content texture.

### Source-Close Direct Objective

If `mean(abs(target_rgb - content_rgb)) <= 0.045`, the sample is treated as
source-close. These are textures whose target pack stays close to vanilla, so
the content/color auxiliary stack can over-constrain the diffusion model. The
trainer uses a direct objective:

```text
total = noise_MSE + 0.50 * recon_L1 + 0.25 * luminance_gradient_L1
```

This route intentionally disables content loss, color moments, hue loss,
contrast gates, and per-channel RGB gradient loss for that sample. It is a
generic target/content-distance heuristic rather than a pack-name special case.
It captures most Bare Bones samples while affecting only a small fraction of
more stylized packs.

### Content-Aware Objective

All other samples use SNR-weighted `x0` auxiliary losses:

```text
total = noise_MSE
      + alpha_bar_t * (0.70 * recon_L1 + 0.30 * gradient_loss + 0.20 * content_loss)
```

`alpha_bar_t` downweights reconstruction and content terms at high noise
timesteps, where the recovered `x0` estimate is less reliable.

### Gradient Loss

- Luminance gradient L1: forward-difference edge preservation on the Y channel.
- Per-channel RGB gradient L1 with weight 0.5: penalizes color bleeding and channel-specific blurring.
- Per-channel RGB gradient loss is part of the content-aware route only.

### Content Loss

Spatially gated, content-relative losses:

| Component | Weight | Description |
|-----------|--------|-------------|
| structure | 1.0 | Gradient delta L1 + 0.5 * detail delta L1 + 0.75 * Laplacian delta L1, all source-relative |
| contrast | 2.5 | Gated under-contrast penalty + 0.6 * edge shortfall |
| hue | 3.0 | Opponent-chroma angular alignment gated by target chroma, plus 0.25 * chroma strength |
| color moment | 1.5 | Per-channel mean/std matching |

Key design decisions:

- Source-relative structure comparisons operate on deltas from the vanilla content rather than raw targets.
- Target-gated contrast and hue losses ignore flat or weakly chromatic regions.
- Contrast is one-sided: it penalizes under-contrast rather than oversharpening.
- Detail emphasis increases penalty on textures with concentrated high-local-variance regions.
- Source-close routing prevents the auxiliary stack from degrading textures where direct diffusion reconstruction is already the right objective.

## Conditioning

- `noisy_target`: current diffusion sample `[B, 3, 32, 32]`.
- `content_rgb`: base/vanilla texture `[B, 3, 32, 32]`.
- `style_refs`: 3 ranked textures from the same target pack `[B, 3, 3, 32, 32]`.
- `style_ref_mask`: boolean validity mask for padded references `[B, 3]`.
- `t`: diffusion timestep `[B]`.

## Inference

### Sampling

- 40 DDPM reverse steps with `x0` clipping to `[0, 1]`.
- Starts from pure Gaussian noise.
- Samples two stochastic diffusion candidates and scores both before routing.

### Dual-Model Candidate Selection

The selector scores candidates with a support-aware structure/style evaluator:

1. Score diffusion candidates for structural quality against the source texture and style quality against support references.
2. For non-wood textures, optionally try detail injection from the recolor candidate at amounts `0.35` and `0.65`.
3. Use RecolorNet fallback when diffusion is not viable or when recolor is clearly better under the selector.
4. Use deterministic support-pair affine recolor or style-stat recolor as a last resort.
5. Include a source-relative style check that compares the candidate's change-from-vanilla magnitude with the support pack's typical change magnitude.

General selector thresholds:

- Minimum structural quality: `0.60`.
- Confident structural quality: `0.80`.
- Clear style margin: `0.10`.

### Wood-Family Routing

Wood textures are detected through `infer_texture_family(texture_id)`. They
receive special routing because plank quality depends on regular grain and
line continuity, and the detail-injection path can damage that structure.

- Diffusion candidates are ranked primarily by style/support agreement.
- Detail injection is skipped.
- RecolorNet fallback is allowed only when diffusion is structurally broken or recolor has a clear style advantage.
- Wood-specific thresholds: minimum structural quality `0.45`, recolor style margin `0.08`, broken-diffusion style tolerance `0.05`.

## What Is Implemented

- Preprocessing filters to full-cube block textures by reading blockstates and model face textures from the base pack.
- Images are resized to `32x32`; `16x` uses nearest-neighbor and `64x` uses Lanczos.
- Validation is filename-based using the fixed set in `config.py`.
- Training samples ranked support textures using vanilla/content descriptors and a filename-family prior from `support_index.py`.
- Runtime debug support for status, snapshots, and preview generation during training.
- `inference/evaluate.py` exists for validation matrices but is not exposed through the CLI.

## Current Gaps / Leftovers

- Alpha channels are preprocessed and loaded, but the model and loss operate on RGB only.
- `64x -> 32x` preprocessing uses Lanczos globally; realistic packs may benefit from a pack-specific or configurable resize policy.
- Support ranking is heuristic rather than learned.
- `inference/export.py` and `config.PALETTE_PATH` are leftovers from the older discrete/palette pipeline and are not part of the current RGB training path.
- Glass and transparent/line-art textures remain difficult and need dedicated handling.
- Complex tiling textures such as leaves and TNT still need stronger structure-aware evaluation.
