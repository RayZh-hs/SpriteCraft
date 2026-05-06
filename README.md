# SpriteCraft

Style-guided Pixel-art Reference Image Transfer Engine for Minecraft

## Installation

Make sure you have [git](https://git-scm.com/) and [uv](https://pypi.org/project/uv/) installed.

```bash
git clone https://github.com/RayZh-hs/SpriteCraft
cd SpriteCraft
uv sync
```

Place a vanilla Minecraft jar file and reference resource packs (do not unzip) in `data/raw_packs/`. All resource packs you use should either be 16x or 32x for best results.

After loading the data, you are encouraged to create a `manifest.json` file in `data/` to specify which resource packs to use and how to use them. Expected format:

```json
{
    "packs": [
        {
            "id": "example_pack_id",
            "archive": "Example Pack.zip",
            "role": "train",    // "train", "base" (jar file), "defer"
            "style": "stylized" // "vanilla", "stylized", "realistic", etc.
        }
    ]
}
```

Remove comments before you continue. The `role` field specifies how the resource pack is used in training. Deferred packs are not used in the training loop. The `style` field is used only in visual evaluation and does not affect training, so you can use any string you like to categorize the style of the resource pack.

## Training Loop

Activate the virtual environment with:

```
source .venv/bin/activate
```

### Preprocessing

This is as simple as running:

```
spritecraft preprocess
```

After this, review the preprocessed data in `data/preprocessed/`. Ensure that enough samples are generated. Somet resource packs overlap sparingly with vanilla block assets, and are not suitable for training.

### Training

Start training with:

```
spritecraft train --checkpoint-dir checkpoints/[RUN_X] --steps [STEPS]
```

This will spin up a training loop. You can use tensorboard to monitor training metrics and visualize training samples. The tensorboard logs are written to `checkpoints/[RUN_X]/tensorboard/`.

```
tensorboard --logdir checkpoints/[RUN_X]/tensorboard
```

### Generation and Evaluation

After a model has been trained, you can use it to generate textures from reference images. Only preprocessed packs can be used for generation, but it need not be part of the training data. You can always add new packs and reprocess without needing to retrain.

Generate specific textures from a target resource pack:

```
spritecraft generate --pack [PACK_ID] --textures stone dirt oak_planks
```

Or randomly sample a number of textures from the base pack and run them through a checkpoint:

```
spritecraft generate --pack [PACK_ID] --random 8
```

Pass `--checkpoint` to point at a checkpoint directory or a specific `.pt` file. Outputs are written under
`output/[PACK_ID]/`. If the target pack contains the queried texture, the output metrics include a
cross-entropy value; otherwise it is omitted.