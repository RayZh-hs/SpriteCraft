# SpriteCraft

Style-guided Pixel-art Reference Image Transfer Engine for Minecraft

---
## Project Structure

```text
SpriteCraft/
├── docs/
│   └── roadmap.md          # Roadmap file
├── data/
│   ├── raw_packs/          # Unzipped packs, only textures/block/
│   └── processed/          # palette.npy, dataset.npz, pair_index.json
├── src/spritecraft/
│   ├── __init__.py
│   ├── __main__.py         # python -m spritecraft
│   ├── cli.py              # CLI entry point (preprocess, train, sample)
│   ├── config.py           # Shared constants and paths
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py   # Filtering, resizing, palette, quantization
│   │   └── dataset.py      # PyTorch dataset / dataloader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py         # U-Net architecture
│   │   └── diffusion.py    # Forward diffusion / masking
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py        # Training loop, CFG, validation
│   └── inference/
│       ├── __init__.py
│       ├── sampler.py      # Iterative decoding
│       └── export.py       # Palette → RGB, resource-pack export
├── checkpoints/
└── output/                 # Generated textures
```

---

## Stage 1: Data Collection

- Get 4 packs: Vanilla (16×), Programmer Art (16×), Faithful 32×, and one different 32× pack.
- Only extract `assets/minecraft/textures/block/`.
- Do not use `item/`, `entity/`, `gui/`, or any JSON model files.

**Pitfall:** Do not collect more than 4 packs. Extra packs add little value but increase preprocessing bugs and imbalance.

---

## Stage 2: Preprocessing

**Filter rules (skip if any fail):**
- File must be `.png` and square.
- Width must be 16 or 32.
- File must not have a `.png.mcmeta` neighbor (skips animations).
- Filename must not contain `_overlay`.
- Image mode is RGBA or RGB.

**Resize:**
- 16× images → upscale to 32× with nearest-neighbor.
- 32× images → keep as-is.

**Handle alpha:**
- If RGBA: composite onto magenta `(255, 0, 255)`, then convert to RGB.
- This makes transparent holes become a distinct color the model can learn.

**Build palette:**
- Collect all pixels from all kept images.
- Run `MiniBatchKMeans` with `k=256`.
- Save centroids as `palette.npy` (shape `(256, 3)`).

**Quantize:**
- Map every 32×32 image to indices `0..255` via nearest centroid.
- Store as `uint8` array `(32, 32)`.

**Build pair index:**
- Key: filename (e.g., `diamond_ore.png`).
- Value: list of `(pack_id, array_index)`.
- Keep only keys present in **3 or more packs**.

**Validation split:**
- Pick 20 common blocks (e.g., `stone`, `dirt`, `planks`).
- Remove them from the training pair index entirely.
- Use only for validation images during training.

**Pitfall:** Skip non-square textures like `redstone_torch.png` (tall sprites), `rail.png` (thin), and `fire_0.png` (animated). They break the fixed 32×32 tensor and have sparse geometry the model cannot reconstruct.

---

## Stage 3: Model Architecture

**Input channels (3 images, each 32×32):**
1. **Noisy target:** the block being generated. Contains `[MASK]` token (index 256).
2. **Content reference:** the source block in original style. Never masked.
3. **Style reference:** one random block from the target pack. Never masked.

Each channel is embedded with `nn.Embedding(257, 128)` and concatenated to 384 channels.

**U-Net:**
- 3 downsample stages: `32 → 16 → 8 → 4`.
- Channel progression: `96 → 192 → 192 → 192`.
- Blocks: ResBlock at each resolution.
- Attention: Only at 8×8 and 4×4. **No attention at 32×32.**
- Time embedding: Sinusoidal + MLP to 128d. Injected into every ResBlock via FiLM (AdaGN).
- Output: `Conv2d` to 256 channels (predicting the 256 palette indices).
- Total parameters: ~20–30M.

**VRAM optimizations:**
- Gradient checkpointing inside ResBlocks.
- `bfloat16` mixed precision.
- `torch.compile(model, mode="reduce-overhead")`.

**Pitfall:** Global attention at 32×32 will cause OOM on an 8GB 4060 Laptop. Keep attention only at low resolutions.

---

## Stage 4: Forward Diffusion

- Type: Absorbing-state (masked) diffusion.
- Timesteps: `T = 50`.
- Mask schedule: `mask_prob(t) = 1 - cos(0.5 * π * (t / T))`.
- At each step: randomly replace pixels in the **target** image with token 256 (`[MASK]`).
- Content and style references are never masked.

**Loss:** `F.cross_entropy(logits, x_clean)` over all 32×32 pixels.

**Pitfall:** Do not apply noise to the content or style reference inputs. Only the target is noised.

---

## Stage 5: Training

- Optimizer: AdamW, lr `1e-4`, weight decay `0.01`.
- Scheduler: Cosine decay to `1e-6`.
- Batch size: 1 or 2.
- Gradient accumulation: 16 steps (effective batch 16).
- Training steps: 100k–200k.

**Classifier-Free Guidance (CFG) training:**
- 10% chance: replace style reference with a blank/zeros image.
- 10% chance: replace content reference with a blank/zeros image.
- This lets you use CFG at inference with no extra training cost.

**Sanity check before full training:**
- Overfit a single (reference, target) pair.
- Loss should drop to near-zero in under 500 steps.
- If it does not, your learning rate is wrong or your architecture has a bug.

**Validation:**
- Every 2k steps, run on the 20 held-out blocks.
- Save side-by-side PNGs: reference → masked → prediction → ground truth.

**Pitfall:** Do not skip the single-pair overfit test. It catches data pipeline and architecture errors before you waste hours on full training.

---

## Stage 6: Inference

**Prepare inputs:**
- User mod texture: resize to 32×32, quantize to palette.
- User style samples: 1–3 textures from target pack, quantize.

**Sampling (iterative decoding):**
1. Start with fully masked target: all pixels = 256.
2. For `t` from `T` down to `1`:
   - Run model: `logits = model(x_t, content_ref, style_ref, t)`.
   - Compute `softmax` probabilities.
   - Pick `confidence = max_prob` per pixel.
   - Determine how many currently-masked pixels to unmask (simple schedule: `ceil(masked_count / t)`).
   - Unmask the highest-confidence pixels. Set them to their predicted indices.
   - Leave already-unmasked pixels unchanged.
3. After step 1, all pixels are unmasked.

**Optional CFG:**
- Run model twice: once conditional, once null (blank references).
- Combine: `logit_uncond + scale * (logit_cond - logit_uncond)`.
- Use scale `2.0`.

**Export:**
- Map indices back to RGB using `palette.npy`.
- If target pack is 16×, downsample output to 16×16 with nearest-neighbor.
- Save as PNG in resource-pack folder structure.

**Pitfall:** Always use `argmax` to get final pixel indices. Never use softmax mean or weighted averaging. Averaging destroys discrete pixel-art structure.

---

## Stage 7: LoRA (Future, Not Planned Now)

The architecture is built so LoRA can be added later without changes.

- Freeze all base model weights.
- Inject rank-4 LoRA matrices into attention projections (`q_proj`, `v_proj`) and FiLM MLPs.
- Train only LoRA weights on ~20–50 paired textures from a new pack.
- Cost: ~5–15 minutes on 4060 Laptop per pack.
- Output: small `.safetensors` file (~500 KB).
- Inference: load base checkpoint + LoRA adapter.

**This stage is reserved for future work. Do not implement it now.**