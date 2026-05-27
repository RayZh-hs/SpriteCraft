# Training Performance Notes

## 2026-05-28: Hue stability and structure fallback

Validation previews from `checkpoints/run26/Ashen_16x` showed two separate failure modes:

- Hue and chroma can drift for many steps because the objective only supervised RGB reconstruction indirectly through diffusion noise and L1 reconstruction. The training CSV already had a partial `content_hue_loss` column from an interrupted edit, but the loss implementation did not emit that component, so the current worktree could fail during training or validation.
- Some high-detail blocks produce support-colored but structurally incoherent samples. `tnt_side.png`, `stone.png`, and `bricks.png` were clear examples where a vanilla-structure recolor is more usable than a noisy diffusion sample.

Changes made in this slice:

- Added an opponent-chroma hue alignment loss in `src/spritecraft/training/content_loss.py`. It avoids HSV wraparound and gates the penalty by target chroma so neutral textures are not over-penalized.
- Added `support_content_refs` to dataset batches and generation support loading. This gives inference access to the vanilla-side texture for every styled support texture.
- Added support-pair affine recolor and detail-injection candidates in `src/spritecraft/inference/sampler.py`. The sampler now evaluates diffusion samples, detail-injected variants, and a deterministic support-pair recolor.
- Added a recolor fallback margin. If the deterministic recolor is a plausible support-style match, it is preferred over a noisy diffusion candidate, matching the production preference that vanilla-like structure is better than incoherent texture synthesis.
- Penalized irrelevant color/stained-glass support matches in `support_index.py`, so uncolored glass prefers `tinted_glass.png` before colored stained-glass supports.

Verification:

- `uv run python -m compileall src`
- Training smoke test on one real `Ashen_16x` batch with backward pass through `_compute_loss`.
- Generation from `checkpoints/run26` on `Ashen_16x` examples:
  - `bricks.png`: MAE improved from `0.1063` in the first candidate-ranking run to `0.0931` with recolor fallback.
  - `tnt_side.png`: MAE improved from `0.1875` to `0.1284`; the output is coherent and vanilla-structured instead of noisy.
  - `stone.png`: MAE improved from `0.0944` to `0.0462`; the output keeps the simple stone structure.
  - `bookshelf.png`: MAE worsened from `0.0921` to `0.1165`, but the output is coherent and vanilla-like rather than structurally noisy. This is an intentional tradeoff under the stated production preference.
  - `glass.png`: still poor (`0.2761` MAE) because available non-validation supports are mostly colored stained-glass variants. Support ranking now chooses `tinted_glass.png` first, but this case still needs either filename-aware glass handling or better support examples.

Next work:

- Train a fresh run with the hue loss enabled and compare hue convergence against `run26`.
- Add candidate-selection diagnostics to generation metadata so it is clear whether output came from diffusion, detail injection, or recolor fallback.
- Evaluate whether glass and other transparent/line-art textures need a dedicated structure-preserving mode.
