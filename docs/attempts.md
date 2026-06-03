# Training Attempts Log

## Run 24 (Baseline)
- **Config:** Diffusion only, linear schedule, 10,000 steps, base_channels=128
- **Quality:** Good. Diffusion model produces high-quality style transfers.
- **Key textures:**
  - `bookshelf.png`: Good, colorful structure preserved
  - `tnt_side.png`: Good, TNT pattern visible
  - `diamond_ore.png`: Good
  - `oak_planks.png`: Good
  - `stone.png`: Excellent (score 0.953)
- **Notes:** No RecolorNet. All outputs are pure diffusion. Takes ~50 min to train.

## Run 30 (Dual Model)
- **Config:** Diffusion + RecolorNet, linear schedule, 10,000 steps, base_channels=128
- **Quality:** Good. RecolorNet adds color consistency, but we need to build better selection criteria for when to use it.

## Run 31 (Output-quality-driven model selection + training improvements)

### Discriminator
- **Approach:** Post-hoc quality evaluation. Diffusion runs first. Output is assessed for
  structural integrity (gradient alignment, edge fidelity, detail coherence vs content)
  and style consistency (support descriptor score vs target pack references).
- **Decision:** If diffusion is structurally viable (≥ 0.45 structural quality), pick the
  model with better style consistency. If diffusion is garbled, fall back to RecolorNet.
  NO hardcoded routing — all decisions are based on actual output traits.
- **Key traits that trigger recolor fallback:** Low structural quality (< 0.45) indicates
  garbled/smeared output (common on complex textures like TNT, bookshelf).

### Training Improvements
- **EMA** (decay=0.995) on both models, used for validation
- **LR Warmup** (LinearLR 0.1→1.0 over first 5% of steps) + CosineAnnealingLR
- **Adaptive content loss weight:** ramps from 0.10→0.30 over first 20% of training
- **Softer contrast gate:** gamma 2.0→1.5, added gate floor=0.05
- **Higher detail emphasis:** 1.5→2.0

### Results (Ashen_16x, 509 train / 19 val, diffusion 10k steps + recolor 5k steps)
| Texture | Selected | MAE | Comment |
|---------|----------|-----|---------|
| stone | diffusion | 0.058 | ✓ |
| oak_planks | diffusion | 0.079 | ✓ |
| diamond_ore | diffusion | 0.074 | ✓ |
| tnt_side | recolor_net | 0.090 | ✓ Correctly detected diffusion structural weakness |
| bookshelf | diffusion | 0.115 | ✓ Better than old recolor (0.181) |
| oak_leaves | diffusion | 0.238 | Leaf structure is hard, similar to recolor (0.246) |
| glass | diffusion | 0.294 | Glass inherently challenging due to alpha/transparency |

### What worked
- Post-hoc structural quality evaluation correctly identifies failed diffusion outputs
- Style consistency comparison selects the better model when both are viable
- Model source tracking clearly shows which model produced each output
- EMA + warmup improved training stability (validation loss 0.60→0.47 over 10k steps)
- Detail injection further improves diffusion outputs for moderate-complexity textures

### What needs more work
- Glass textures: structurally simple but extreme color variance across packs
- Leaf textures: high structural complexity, both models struggle
- Structural quality threshold needs per-pack calibration for optimal results

## Run 35/36 (Wood content-loss relaxation probe)

### Motivation
- **Problem:** Compared with `run20`, newer diffusion outputs in `run33` show a
  small but systematic downgrade on wood/plank textures, especially
  Bare_Bones_1_21_11 `oak_log.png` and `oak_planks.png`.
- **Hypothesis:** The newer content/color-aware auxiliary losses over-constrain
  wood textures by preserving vanilla grain/detail when some target packs
  intentionally simplify or redraw planks/logs.
- **Note:** The current model is continuous RGB diffusion, so there is no
  cross-entropy objective in the active path. The closest fallback is reducing
  auxiliary x0/content terms and relying more heavily on noise MSE plus
  reconstruction/gradient terms.

### Change Tested
- Added a per-sample `content_loss_scale` in `training/train.py`.
- Generic gate lowers content-aware losses when target-vs-vanilla
  gradient/detail agreement is low or the target is visibly simplified.
- Wood-family textures are capped at `0.35`.
- Validation logging records `content_loss_scale`.
- `run36` only fixes validation CSV fieldnames for the new scalar.

### Scale Diagnostic
| Pack | Texture | content_loss_scale |
|------|---------|--------------------|
| Bare_Bones_1_21_11 | oak_log.png | 0.350 |
| Bare_Bones_1_21_11 | oak_planks.png | 0.350 |
| Bare_Bones_1_21_11 | spruce_planks.png | 0.350 |
| Bare_Bones_1_21_11 | stone.png | 0.919 |
| Bare_Bones_1_21_11 | tnt_side.png | 1.000 |
| Chibli_64x_Freepack | oak_planks.png | 0.201 |
| Chibli_64x_Freepack | spruce_planks.png | 0.244 |

### Focused Probe
- **Pack:** Bare_Bones_1_21_11
- **Start checkpoint:** `checkpoints/run33/Bare_Bones_1_21_11`
- **Probe checkpoint dir:** `checkpoints/run35_finetune/Bare_Bones_1_21_11`
- **Meaningful checkpoint:** step 10,500. The later 11,000-step probe is not
  reliable because changing the target step count caused the scheduler to
  reset and jump LR back to ~3e-4 after step 10,550.

| Texture | run20 diffusion | run33 diffusion | run35 probe step 10,500 |
|---------|-----------------|-----------------|--------------------------|
| oak_log.png | MAE 0.0136 / acc 0.9840 | MAE 0.0418 / acc 0.6283 | MAE 0.0396 / acc 0.6478 |
| oak_planks.png | MAE 0.0090 / acc 0.9922 | MAE 0.0321 / acc 0.8646 | MAE 0.0359 / acc 0.7773 |
| spruce_planks.png | MAE 0.0253 / acc 0.9613 | MAE 0.0238 / acc 0.9707 | MAE 0.0258 / acc 0.9639 |

### Result
- **Mixed / not successful.**
- Relaxing the content loss slightly improves Bare Bones `oak_log.png`, keeps
  `spruce_planks.png` close to run33, but worsens `oak_planks.png`.
- Visual inspection confirms that step 10,500 does not recover the near-perfect
  Bare Bones plank structure seen in `run20`.

### Takeaway
- The Bare Bones plank regression is probably not caused solely by the
  content/color-aware auxiliary loss magnitude.
- Next useful ablation should separate:
  1. `run33` loss as-is,
  2. wood content loss disabled,
  3. wood hue/color moment losses disabled,
  4. broader wood x0 auxiliary relaxation while keeping noise MSE and direct
     reconstruction.
