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

## Run 31 (Smart Discriminator + Training Improvements)
- **Config:** Diffusion 10k steps + RecolorNet 5k steps, cosine schedule, base_channels=128
- **Training improvements:** EMA (decay=0.995), LR warmup (5%), adaptive content loss weight (ramp 0.10→0.30), softer contrast gate (gamma 2.0→1.5, floor=0.05), higher detail emphasis (1.5→2.0)
- **Discriminator:** Pre-classification texture complexity classifier using entropy, FFT, gradients, family heuristics. Routes "simple" to diffusion, "complex" to recolor, "moderate" to diffusion.
- **Quality:** Excellent. Classifier correctly routes bookshelf/tnt/leaves→recolor, ores/wood/stone→diffusion.
- **Key textures (Ashen_16x):**
  - `dirt.png`: mae=0.061 ✓ (diffusion)
  - `oak_log.png`: mae=0.061 ✓ (diffusion)
  - `oak_planks.png`: mae=0.062 ✓ (diffusion)
  - `gravel.png`: mae=0.069 ✓ (diffusion)
  - `stone.png`: mae=0.082 ✓ (diffusion)
  - `diamond_ore.png`: mae=0.074 ✓ (diffusion)
  - `tnt_side.png`: mae=0.090 ✓ (recolor)
  - `cobblestone.png`: mae=0.092 ✓ (diffusion)
  - `bookshelf.png`: mae=0.181 ✓ (recolor)
  - `glass.png`: mae=0.309 (diffusion - glass is inherently hard due to transparency/alpha)
  - `oak_leaves.png`: mae=0.246 ✓ (recolor)
- **Model source tracking:** Each output reports classifier verdict, selected model source, and model score.
- **Notes:** Glass textures remain challenging - structurally simple but extreme color variance across packs. Diffusion handles glass better than recolor (MAE 0.31 vs 0.37). Overall training time ~55 min for diffusion, ~5 min for recolor on Ashen_16x (509 train textures).
