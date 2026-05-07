# Diffusion Refactor Notes

## Problem Summary

- Simple, low-frequency tiles work because the model can regress their average structure from vanilla content plus a weak style hint.
- High-contrast, high-frequency tiles fail because edges, mortar lines, shelf boundaries, and fine texture are either averaged away or replaced with stochastic noise.
- The issue is strongest on realistic packs because the current `32x32` RGB target compresses more detail, while support selection and the objective do not strongly preserve local structure.

## Main Root Causes

1. `StyleAwareUNet.forward()` ignores `noisy_target`.
   The model is trained inside a diffusion loop, but it does not condition on the current noisy sample. In effect, it behaves like a direct conditional regressor with a timestep embedding attached. This makes sampling mathematically mismatched and unstable on detailed textures.

2. The objective favors averages.
   `L1 + 0.1 *` simplified global SSIM in [src/spritecraft/training/train.py](/home/rayzh/Projects/SpriteCraft/src/spritecraft/training/train.py) is enough for flat planks, but it does not strongly punish blurred edges or misplaced local contrast.

3. Style evidence is weak and noisy.
   Training uses random support textures; generation is effectively 1-shot by default in [src/spritecraft/inference/generate.py](/home/rayzh/Projects/SpriteCraft/src/spritecraft/inference/generate.py). On complex packs, one unrelated support image is not enough to recover pack-specific microtexture.

4. Style references are over-compressed.
   The current model encodes all support textures and then mean-pools them into one `8x8` latent in [src/spritecraft/models/unet.py](/home/rayzh/Projects/SpriteCraft/src/spritecraft/models/unet.py). Averaging support features is exactly what destroys crisp, high-contrast motifs.

5. Sampling and training are inconsistent.
   Training predicts clean RGB directly from arbitrary timesteps; inference iteratively injects new noise anyway. This mismatch is likely contributing visible grain and instability.

6. Preprocessing loses detail on high-res packs.
   `64x -> 32x` Lanczos downsampling in [src/spritecraft/data/preprocess.py](/home/rayzh/Projects/SpriteCraft/src/spritecraft/data/preprocess.py) can wash out pixel-art-like edges or fold high-frequency texture into mush before training even starts.

## Recommended Fixes

### Phase 1: Fix the biggest correctness issues first

- Make the network truly conditional on `noisy_target`.
  Add a dedicated encoder path for `noisy_target` and fuse it with content/style features at multiple scales. Without this, the model is not really denoising.

- Stop using random 1-shot support at inference.
  Use `support_index.py` ranking and default to `3-shot` generation when available. Support quality is a first-order factor for bookshelves, bricks, ores, and realistic packs.

- Replace support mean-pooling with token-level attention.
  Flatten each support texture into tokens and let content/noisy features attend across all reference tokens, optionally with a learned support mask. Do not collapse all supports into one average feature map.

- Use a sharper reconstruction objective.
  Keep `L1`, but add a local gradient or Laplacian loss on luminance. This directly penalizes blurry mortar lines and missing shelf edges.

### Phase 2: Simplify or fix the diffusion formulation

- Pick one path and commit to it:
  - Preferred short-term path: drop diffusion entirely and train a deterministic conditional generator first.
  - Preferred long-term path: keep diffusion, but train on `epsilon` or `v` prediction and use a standard sampler consistently.

- If diffusion stays:
  - Predict `epsilon`/`v`, not clean RGB.
  - Pass `noisy_target` through the network at every step.
  - Use the same forward objective and reverse sampler family in both training and inference.
  - Increase timesteps only after correctness is fixed; `20` steps is not the main bottleneck right now.

## Data and Resolution Improvements

- Treat realistic packs separately from simple packs.
  The same `32x32` target is too restrictive for some `64x` sources. Add a `64x64` training mode or at least evaluate it for high-detail packs.

- Revisit downsampling policy.
  Lanczos is good for photography, but not always for stylized textures. Compare Lanczos vs bicubic vs area for `64x -> 32x` pack preprocessing.

- Train with texture-family-aware support sampling.
  Prefer supports from the same family and surface role (`brick`, `stone`, `ore`, `_top`, `_side`, etc.). The ranking helpers already point in this direction.

## Suggested Execution Plan

1. Implement support ranking in both training and generation; raise default support count from `1` to `3`.
2. Add gradient/Laplacian loss and re-run validation on `bricks.png` and `bookshelf.png`.
3. Refactor `StyleAwareUNet` so `noisy_target` is encoded and fused at multiple scales.
4. Replace support mean-pooling with token-level multi-reference attention.
5. Decide whether to keep diffusion:
   If results improve sharply after steps 1 to 4, convert to proper epsilon/v diffusion; otherwise, replace diffusion with a direct conditional generator and use that as the baseline.

## Expected Outcome

- Bricks should regain straighter mortar boundaries and less chroma noise.
- Bookshelves should stop smearing book spines into the wood frame.
- Realistic packs should become less grainy because support choice and local-structure loss will constrain texture synthesis much more tightly.
