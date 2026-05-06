# Refactor: Sprite Inclusion in Training and Inference

Currently, only block textures are used to train the model, and sprites are not included in the training data.

Sprites are also an important part of the Minecraft world, and there are many more sprites than feasible block textures. They should be included in the training data (in preprocessing part) and inference. The current validation logic should be altered to accommodate them. 