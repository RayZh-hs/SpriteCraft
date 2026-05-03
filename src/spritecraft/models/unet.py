"""U-Net model architecture."""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int = 128):
        super().__init__()
        # TODO: Implement ResBlock
        pass

    def forward(self, x, cond):
        raise NotImplementedError("ResBlock forward not yet implemented")


class UNet(nn.Module):
    """U-Net for masked diffusion on pixel-art textures."""

    def __init__(self, vocab_size: int = 257, embed_dim: int = 128):
        super().__init__()
        # TODO: Implement U-Net architecture
        pass

    def forward(self, noisy_target, content_ref, style_ref, t):
        raise NotImplementedError("UNet forward not yet implemented")
