# SpriteCraft Poster Content Plan

## Format

- Size: A0 portrait, 841mm x 1189mm
- Layout: two content columns, rows=30
- Design: standard SJTU-style academic poster template: purple header/footer, white body, large purple section headings, thin divider rules, no large colored content cards
- Source material: `report/main.tex`, `report/chapters/*.tex`, `report/figures/*.png`, `output/run33_validation/summary.json`, and `poster/CS3308-ML-poster-template.pptx`
- Image policy: use only report/checkpoint Minecraft texture images as supporting figures; no generated images required

## Narrative

The poster presents SpriteCraft as a practical per-pack Minecraft texture style-transfer system. The main story is:

1. Vanilla block textures are paired with resource-pack targets after full-cube filtering and 32x32 normalization.
2. A style-conditioned diffusion model provides expressive generation, while RecolorNet preserves structure.
3. Output-driven routing selects diffusion, detail-injected blends, or recolor fallback depending on measured structural and style quality.
4. Results are pack-dependent: minimalist packs are easiest, Ashen improves with the final configuration, and Chibli/glass/fine symbolic textures remain hard.

## Top Span

- Purple title bar modeled after `CS3308-ML-poster-template.pptx`
- Title: SpriteCraft: Texture Style Transfer for Minecraft Resource Packs
- Subtitle: Diffusion generation with structure-preserving recolor fallback for 32x32 block textures
- Author: Ray Zhang, Shanghai Jiao Tong University
- SJTU wordmark extracted from the template as `poster/figures/sjtu-logo-white.png`

## Left Column

- Background
  - Manual resource-pack creation is labor-intensive.
  - Small pixel textures amplify color and edge errors.
  - Vanilla filenames provide paired supervision across packs.

- Task
  - Figure: `spritecraft-pipeline-overview.png`
  - Four compact facts: 32x32 grid, three ranked style references, 20 validation textures, four fully trained run33 packs.

- Model
  - StyleAwareUNet predicts diffusion noise conditioned on content, style references, and timestep.
  - RecolorNet provides deterministic residual recoloring with structure preservation.
  - Figure: `spritecraft-dual-model-architecture.png`

- Training objective
  - SNR-weighted x0 reconstruction and content losses.
  - Source-relative structure, contrast, hue, and color-moment losses.

## Right Column

- Our Approach
  - Candidate routing via structural quality and style quality.
  - Figure: `spritecraft-ensemble-routing-flow.png`

- Experiments
  - Table: run33 MAE / pixel accuracy for diffusion, recolor, and selected outputs.

- Results
  - Figure: `run33-multi-pack-oak-planks.png`
  - Pack-dependent findings: Bare Bones easiest, Ashen improved, Chibli hardest.

- Challenges
  - Diffusion is useful when structure is safe.
  - Recolor fallback wins on high-pattern textures.
  - Figure: `run33-ashen-routing.png`
  - Future work: alpha-aware training, learned support selection, multi-resolution refinement, learned routing thresholds.

## Figure Selection

- `spritecraft-pipeline-overview.png`: task/pipeline support figure
- `spritecraft-dual-model-architecture.png`: model support figure
- `spritecraft-ensemble-routing-flow.png`: approach/routing support figure
- `run33-multi-pack-oak-planks.png`: qualitative result support figure
- `run33-ashen-routing.png`: challenge/stress-test support figure

## Approximate Word Count

- Body text: about 430 words
- Figure captions and table text: about 90 words
- Total visible prose: about 520 words

