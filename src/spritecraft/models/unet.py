"""U-Net model architecture."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spritecraft.config import IMAGE_SIZE, NUM_TIMESTEPS, PALETTE_SIZE, VOCAB_SIZE


def _group_norm_groups(num_channels: int, max_groups: int = 32) -> int:
    """Choose the largest valid GroupNorm group count up to max_groups."""
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    if half == 0:
        raise ValueError("Embedding dimension must be at least 2")

    exponent = -math.log(max_period) * torch.arange(
        half,
        device=timesteps.device,
        dtype=torch.float32,
    ) / half
    freqs = torch.exp(exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


def _apply_film(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply FiLM modulation to a feature map."""
    return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class Downsample(nn.Module):
    """Strided convolution downsampler."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsampler followed by a 3x3 projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Self-attention over spatial tokens."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_group_norm_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        head_dim = channels // self.num_heads

        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(batch, self.num_heads, head_dim, height * width).transpose(2, 3)
        k = k.reshape(batch, self.num_heads, head_dim, height * width).transpose(2, 3)
        v = v.reshape(batch, self.num_heads, head_dim, height * width).transpose(2, 3)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(2, 3).reshape(batch, channels, height, width)
        return x + self.proj_out(attn)


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 128,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_channels), out_channels, affine=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _forward_impl(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.cond_proj(F.silu(cond)).chunk(2, dim=1)
        h = self.norm2(h)
        h = _apply_film(h, scale, shift)
        h = self.conv2(F.silu(h))

        return self.skip(x) + h

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            ckpt = checkpoint(self._forward_impl, x, cond, use_reentrant=False)
            assert isinstance(ckpt, torch.Tensor)
            return ckpt
        return self._forward_impl(x, cond)


class UNet(nn.Module):
    """U-Net for masked diffusion on pixel-art textures."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 128,
        cond_dim: int = 128,
        base_channels: int = 96,
        num_timesteps: int = NUM_TIMESTEPS,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.cond_dim = cond_dim
        self.num_timesteps = num_timesteps

        ch32 = base_channels
        ch16 = base_channels * 2
        ch8 = base_channels * 2
        ch4 = base_channels * 2

        self.target_embed = nn.Embedding(vocab_size, embed_dim)
        self.content_embed = nn.Embedding(vocab_size, embed_dim)
        self.style_embed = nn.Embedding(vocab_size, embed_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

        self.stem = nn.Conv2d(embed_dim * 3, ch32, kernel_size=3, padding=1)

        self.enc32 = ResBlock(ch32, ch32, cond_dim=cond_dim, use_checkpoint=use_checkpoint)
        self.down32_16 = Downsample(ch32, ch16)

        self.enc16 = ResBlock(ch16, ch16, cond_dim=cond_dim, use_checkpoint=use_checkpoint)
        self.down16_8 = Downsample(ch16, ch8)

        self.enc8 = ResBlock(ch8, ch8, cond_dim=cond_dim, use_checkpoint=use_checkpoint)
        self.attn8 = AttentionBlock(ch8)
        self.down8_4 = Downsample(ch8, ch4)

        self.enc4 = ResBlock(ch4, ch4, cond_dim=cond_dim, use_checkpoint=use_checkpoint)
        self.attn4 = AttentionBlock(ch4)

        self.mid1 = ResBlock(ch4, ch4, cond_dim=cond_dim, use_checkpoint=use_checkpoint)
        self.mid_attn = AttentionBlock(ch4)
        self.mid2 = ResBlock(ch4, ch4, cond_dim=cond_dim, use_checkpoint=use_checkpoint)

        self.up4_8 = Upsample(ch4, ch8)
        self.dec8 = ResBlock(ch8 + ch8, ch8, cond_dim=cond_dim, use_checkpoint=use_checkpoint)
        self.dec8_attn = AttentionBlock(ch8)

        self.up8_16 = Upsample(ch8, ch16)
        self.dec16 = ResBlock(ch16 + ch16, ch16, cond_dim=cond_dim, use_checkpoint=use_checkpoint)

        self.up16_32 = Upsample(ch16, ch32)
        self.dec32 = ResBlock(ch32 + ch32, ch32, cond_dim=cond_dim, use_checkpoint=use_checkpoint)

        self.out_norm = nn.GroupNorm(_group_norm_groups(ch32), ch32)
        self.out_conv = nn.Conv2d(ch32, PALETTE_SIZE, kernel_size=3, padding=1)

    def _validate_inputs(
        self,
        noisy_target: torch.Tensor,
        content_ref: torch.Tensor,
        style_ref: torch.Tensor,
        t: torch.Tensor,
    ) -> None:
        if noisy_target.ndim != 3 or content_ref.ndim != 3 or style_ref.ndim != 3:
            raise ValueError("Expected image tensors with shape (batch, height, width)")

        expected_hw = (IMAGE_SIZE, IMAGE_SIZE)
        if noisy_target.shape[1:] != expected_hw:
            raise ValueError(f"Expected noisy_target shape (*, {IMAGE_SIZE}, {IMAGE_SIZE}), got {tuple(noisy_target.shape)}")
        if content_ref.shape != noisy_target.shape or style_ref.shape != noisy_target.shape:
            raise ValueError("Target, content reference, and style reference must have matching shapes")

        if t.ndim != 1:
            raise ValueError(f"Expected timestep tensor with shape (batch,), got {tuple(t.shape)}")
        if t.shape[0] != noisy_target.shape[0]:
            raise ValueError("Timestep batch dimension must match image batch dimension")

        if torch.any(t < 0) or torch.any(t > self.num_timesteps):
            raise ValueError(f"Timesteps must be within [0, {self.num_timesteps}]")

    def forward(
        self,
        noisy_target: torch.Tensor,
        content_ref: torch.Tensor,
        style_ref: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(noisy_target, content_ref, style_ref, t)

        noisy_target = noisy_target.long()
        content_ref = content_ref.long()
        style_ref = style_ref.long()
        t = t.to(device=noisy_target.device, dtype=torch.long)

        target_tokens = self.target_embed(noisy_target).permute(0, 3, 1, 2)
        content_tokens = self.content_embed(content_ref).permute(0, 3, 1, 2)
        style_tokens = self.style_embed(style_ref).permute(0, 3, 1, 2)

        x = torch.cat([target_tokens, content_tokens, style_tokens], dim=1)
        cond = self.time_mlp(_timestep_embedding(t, self.cond_dim))

        x32 = self.enc32(self.stem(x), cond)
        x16 = self.enc16(self.down32_16(x32), cond)

        x8 = self.enc8(self.down16_8(x16), cond)
        x8 = self.attn8(x8)

        x4 = self.enc4(self.down8_4(x8), cond)
        x4 = self.attn4(x4)

        h = self.mid1(x4, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)

        h = self.up4_8(h)
        h = self.dec8(torch.cat([h, x8], dim=1), cond)
        h = self.dec8_attn(h)

        h = self.up8_16(h)
        h = self.dec16(torch.cat([h, x16], dim=1), cond)

        h = self.up16_32(h)
        h = self.dec32(torch.cat([h, x32], dim=1), cond)

        return self.out_conv(F.silu(self.out_norm(h)))
