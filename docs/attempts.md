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
- **Plan:** 
  1. Build a texture complexity classifier that pre-classifies textures as "diffusion-friendly" vs "recolor-required" based on content features (entropy, FFT, gradients, family).
  2. Integrate the classifier into the inference pipeline to skip wasteful diffusion attempts on complex textures.
  3. Report which model produced each output.
  4. Improve training: EMA, warmup LR, adaptive loss weights, better content loss gating.
  5. Train both models and validate with generation.
- **Status:** In progress
