# SpriteCraft Poster Content Plan

## Format

- Size: A0 portrait, 841mm x 1189mm
- Layout: two content columns, rows=24
- Design: dark pixel-art/material palette with tinted cards and left accent stripes
- Source material: `report/main.tex`, `report/chapters/*.tex`, `report/figures/*.png`, and `output/run33_validation/summary.json`
- Image policy: use only report/checkpoint Minecraft texture images; no generated images required

## Narrative

The poster presents SpriteCraft as a practical per-pack Minecraft texture style-transfer system. The main story is:

1. Vanilla block textures are paired with resource-pack targets after full-cube filtering and 32x32 normalization.
2. A style-conditioned diffusion model provides expressive generation, while RecolorNet preserves structure.
3. Output-driven routing selects diffusion, detail-injected blends, or recolor fallback depending on measured structural and style quality.
4. Results are pack-dependent: minimalist packs are easiest, Ashen improves with the final configuration, and Chibli/glass/fine symbolic textures remain hard.

## Headline Stats

- 32x32 RGB: canonical training and inference resolution
- 7.7M + 1.0M: StyleAwareUNet and RecolorNet parameter counts
- 40 steps: cosine-schedule DDPM sampling
- 4 packs: run33 fully trained diffusion validation packs

## Two-Column Layout

### Top Span

- Title: SpriteCraft: Texture Style Transfer for Minecraft Resource Packs
- Subtitle: Diffusion generation + structure-preserving recolor fallback for 32x32 block textures
- Author: Ray Zhang, Shanghai Jiao Tong University
- Stat banner with four compact facts listed above

### Left Column

- Motivation and data framing
  - Manual resource-pack creation is labor-intensive.
  - Small pixel textures amplify color and edge errors.
  - Vanilla filenames provide paired supervision across packs.
  - Figure: `spritecraft-pipeline-overview.png`

- Dual-model method
  - StyleAwareUNet predicts diffusion noise conditioned on content, style references, and timestep.
  - RecolorNet provides deterministic residual recoloring with structure preservation.
  - Figure: `spritecraft-dual-model-architecture.png`

- Training objective and inference rules
  - SNR-weighted x0 reconstruction and content losses.
  - Source-relative structure, contrast, hue, and color-moment losses.
  - Candidate routing via structural quality and style quality.

### Right Column

- Cross-pack results
  - Table: run33 MAE / pixel accuracy for diffusion, recolor, and selected outputs.
  - Figure: `run33-multi-pack-oak-planks.png`

- Selector behavior
  - Diffusion is useful when structure is safe.
  - Recolor fallback wins on high-pattern textures.
  - Non-destructive detail injection helps easy packs retain detail.
  - Figure: `run33-ashen-routing.png`

- What works / what fails
  - Works: vanilla-adjacent, minimalist, and medium-complexity stylized packs.
  - Fails: alpha materials, high-resolution microtexture compressed to 32x32, fine symbolic detail.

- Takeaways
  - A dual-model ensemble is safer than diffusion alone.
  - Quality-driven routing is more useful than fixed texture-level rules.
  - Future work: alpha-aware training, learned support selection, multi-resolution refinement, learned routing.

## Figure Selection

- `spritecraft-pipeline-overview.png`: left column, data pipeline card
- `spritecraft-dual-model-architecture.png`: left column, method card
- `run33-multi-pack-oak-planks.png`: right column, result card
- `run33-ashen-routing.png`: right column, stress-test/routing card

## Approximate Word Count

- Body text: about 390 words
- Figure captions and table text: about 90 words
- Total visible prose: about 480 words

