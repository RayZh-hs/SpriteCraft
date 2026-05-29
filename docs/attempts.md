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
