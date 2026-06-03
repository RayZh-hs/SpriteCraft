# SpriteCraft Report Outline

## 1. Introduction

- Texture transfer has been a significant area of research in computer graphics, and diffusion models have shown great promise in this domain.
- However, modern diffusion models are large by nature, and are mostly trained on general image datasets (#reference). This makes them less effective for pixel art texture synthesis.
- We propose SpriteCraft, a novel diffusion-based approach for texture style transfer in Minecraft resource packs.
- This doc...

## 2. Background

...

## 3. Methodology

(Present both "what failed" and "what worked".)

Takeaways:
- Discrete diffusion is less effective in Minecraft tex gen, possibly also in general pixel art, where large contrast in color is vastly common, and models take much longer to understand "closeness" of colors and what to use where.
- Continuous diffusion is more effective, but still has issues with color drifting (especially in early stages of training, where hues drift drastically, even after introducing a hue loss term).
- "No Free Lunch" stands: It is much easier to train a model to learn a specific style than to learn generic style transfer. The goal of this project is not to provide a general-purpose checkpoint model that can be used for arbitrary style transfer, but to provide a framework to rapidly train models to transfer specific styles.
- Loss functions are most potent in driving the "average quality" up, but can backfire in textures that are easiest to learn, causing the loss curve to drop much slower. Contrast the Bare Bones diffusion-only baseline with the current source-close routed objective. (Better: Generate a new test set trained on Bare Bones with and without auxiliary losses for a cleaner comparison.) This may be due to the fact that the various criteria added by penalty terms are rules of thumb, interfering with the model's learning process for tasks it otherwise already excels at.
- Zero shot is hard: We need "weaker reference" -> learn to color transfer.

Finalized network status (with detailed image: output prompt for me to feed to GPT-Image2 to obtain)

Read the refactoring process in the git worktree for details and fledge this section out.

## 4. Experiments

Analyze the process of preprocessing ~ training ~ ensemble. Analyze the results supported by current checkpoint images and semantic report figures.

## 5. Conclusion

What this entails.

Future work includes generic style transfer, and automatic routing logic in the ensemble.

## Appendix: Reproduction Instructions

Brief introduction to using the codebase.
