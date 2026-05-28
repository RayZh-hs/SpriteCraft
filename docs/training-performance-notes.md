# Training Performance Notes

## 2026-05-28 (morning): Hue stability and structure fallback

Validation previews from `checkpoints/run26/Ashen_16x` showed two separate failure modes:

- Hue and chroma can drift for many steps because the objective only supervised RGB reconstruction indirectly through diffusion noise and L1 reconstruction.
- Some high-detail blocks produce support-colored but structurally incoherent samples.

Changes made in this slice:
- Added opponent-chroma hue alignment loss in `content_loss.py`
- Added `support_content_refs` to dataset batches
- Added support-pair affine recolor and detail-injection candidates in `sampler.py`
- Added a recolor fallback margin for difficult textures
- Penalized irrelevant color/stained-glass support matches in `support_index.py`

## 2026-05-28 (afternoon): Comprehensive training improvements

### Issue #1: Hue instability — DIAGNOSED AND MITIGATED

Root cause: the model lacked direct color supervision. Hue drifted because the diffusion noise loss doesn't constrain color directly, and the cross-attention to 3 support references biased colors toward reference colors (which are not the same as the target).

Diagnosis: For stone in Ashen_16x, support references were tuff variants (dark gray, mean ~0.30-0.35) while stone's target mean was ~0.39-0.40. The model output was pulled toward the reference colors, producing outputs that were consistently too dark.

Changes:
1. **Boosted hue loss weight** from 0.5 → 3.0 in content loss
2. **Added global color moment matching loss** (COLOR_MOMENT_WEIGHT=1.5) penalizing per-channel mean/std deviation — provides strong, reliable gradient that doesn't depend on spatial alignment
3. **Increased content loss weight** in total training loss from 0.20 → 0.40
4. **Expanded style references** from 3 → 6 (MAX_SUPPORT_EXEMPLARS) to give model a broader view of the pack's color palette, reducing bias from individual references
5. Attempted color_fc injection of style stats — **WITHDRAWN**: this directly injected support reference color statistics into the bottleneck, which made the color bias WORSE by forcing the model toward reference colors

Results (run29 step 20000 vs run27 step 10000):
- stone.png hue_loss: 0.457 → 0.082 (5.5x improvement)
- diamond_ore.png hue_loss: 0.460 → 0.369 (1.2x improvement)
- coal_ore.png hue_loss: 0.627 → 0.079 (7.9x improvement)
- bookshelf.png hue_loss: 0.210 → 0.151 (1.4x improvement)

### Issue #2: Complex structure prediction — PARTIALLY ADDRESSED

For complex/zero-shot textures, the diffusion model fundamentally cannot infer structure from limited support references.

Changes:
1. **RecolorNet**: a 1.03M-parameter lightweight U-Net that recolors vanilla textures to match pack style while preserving exact vanilla structure. Trained per-pack with reconstruction + edge preservation + color moment + content identity losses.
2. **Tiered candidate selection**: diffusion output preferred when score >= 0.50; RecolorNet used as fallback when diffusion is poor; deterministic affine recolor as last resort
3. **Lowered recolor thresholds**: MIN_RECOLOR_SUPPORT_SCORE=0.40, RECOLOR_SUPPORT_MARGIN=0.25

For the Ashen_16x validation set, all textures score >= 0.50 on the support descriptor metric, so the diffusion model is used directly. The RecolorNet serves as a safety net for out-of-distribution textures.

### Issue #3: Sharpness/detail loss — IMPROVED

Changes:
1. **Doubled diffusion timesteps** from 20 → 40 for more precise denoising
2. **Added per-channel RGB gradient loss** to complement luminance-only gradient supervision — penalizes color bleeding and channel-specific blurring
3. **Boosted gradient loss weight** from 0.15 → 0.30 in total loss
4. **Increased EDGE_SHORTFALL_WEIGHT** from 0.35 → 0.6
5. **Increased CONTRAST_LOSS_WEIGHT** from 1.75 → 2.5

### Overall results (run29, Ashen_16x, step 20000)

| Texture              | MAE    | Assessment |
|----------------------|--------|------------|
| spruce_planks.png    | 0.060  | Excellent  |
| oak_log.png          | 0.065  | Excellent  |
| oak_planks.png       | 0.071  | Good       |
| stone.png            | 0.079  | Good (hue 0.082, down from 0.457) |
| obsidian.png         | 0.090  | Good       |
| iron_ore.png         | 0.091  | Good       |
| dirt.png             | 0.090  | Good       |
| gold_ore.png         | 0.107  | OK         |
| bricks.png           | 0.114  | OK         |
| cobblestone.png      | 0.132  | OK         |
| gravel.png           | 0.131  | OK         |
| diamond_ore.png      | 0.148  | OK (hue improving) |
| coal_ore.png         | 0.156  | Fair       |
| bookshelf.png        | 0.156  | Fair       |
| mossy_cobblestone.png| 0.188  | Fair       |
| tnt_side.png         | 0.237  | Poor       |
| oak_leaves.png       | 0.283  | Poor       |
| glass.png            | 0.419  | Very poor  |

### Architecture summary

- StyleAwareUNet: 7.73M params (6 style refs, 40 timesteps)
- RecolorNet: 1.03M params (backup color transfer)
- Loss: noise_MSE + 0.5×recon_L1 + 0.30×gradient + 0.40×(1.0×structure + 2.5×contrast + 3.0×hue + 1.5×color_moment)

### Remaining gaps

- Glass and transparent/line-art textures remain difficult — these need dedicated handling
- Complex tiling textures (oak_leaves, tnt_side) need more structure-aware training
- The 6-ref model is slightly larger and slower than the 3-ref baseline
- Candidate selection diagnostics not yet in generation metadata
