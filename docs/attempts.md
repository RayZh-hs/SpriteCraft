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
