"""U-Net model architecture with pack embeddings."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentEncoder(nn.Module):
    """Extracts spatial structure from a source exemplar."""

    def __init__(self, vocab_size: int = 257, embed_dim: int = 192, out_dim: int = 256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, out_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 128)
        self.norm2 = nn.GroupNorm(8, 128)

    def forward(self, source_tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(source_tokens.long())  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        return x  # [B, out_dim, H, W]


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int = 256):
        super().__init__()
        assert channels % 32 == 0, "Channels must be divisible by 32 for GroupNorm"
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, channels * 4)
        self.activation = nn.SiLU()

    def _get_film_tr(self, cond: torch.Tensor):
        """Compute FiLM transformation, generating [batch, channels, 1, 1] tensors"""
        scale1, shift1, scale2, shift2 = self.cond_proj(cond).chunk(4, dim=-1)
        scale1 = scale1[:, :, None, None]
        shift1 = shift1[:, :, None, None]
        scale2 = scale2[:, :, None, None]
        shift2 = shift2[:, :, None, None]
        return scale1, shift1, scale2, shift2

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        residual = x
        scale1, shift1, scale2, shift2 = self._get_film_tr(cond)

        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h = self.activation(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = h * (1 + scale2) + shift2
        h = self.activation(h)
        h = self.conv2(h)

        return h + residual


class SelfAttention(nn.Module):
    """Spatial self-attention with GroupNorm and residual connection."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)

        h = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        h = h.transpose(2, 3).reshape(B, C, H, W)
        h = self.proj(h)
        return x + h


class UNet(nn.Module):
    """U-Net for masked diffusion on pixel-art textures with pack embeddings."""

    class Downsample(nn.Module):
        """Downsampling block"""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    class Upsample(nn.Module):
        """Upsampling block"""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor):
            x = self.upsample(x)
            return self.conv(x)

    def __init__(self, num_packs: int, vocab_size: int = 257,
                 style_dim: int = 256, embed_dim: int = 192):
        super().__init__()
        self.embed_dim = embed_dim
        self.style_dim = style_dim

        # Content structure: extract from source exemplar
        self.content_encoder = ContentEncoder(vocab_size, embed_dim, style_dim)

        # Pack style: what aesthetic to apply?
        self.pack_embedding = nn.Embedding(num_packs, style_dim)

        # Time embedding: where in the diffusion process?
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, style_dim)
        )

        # Token embedding for the noisy target (unchanged)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Input projection: noisy tokens + content structure
        self.in_proj = nn.Conv2d(embed_dim + style_dim, 128, kernel_size=3, padding=1)

        self.enc32 = ResBlock(128, cond_dim=style_dim)
        self.down32_16 = self.Downsample(128, 256)
        self.enc16 = ResBlock(256, cond_dim=style_dim)
        self.down16_8 = self.Downsample(256, 384)
        self.enc8 = ResBlock(384, cond_dim=style_dim)
        self.attn8 = SelfAttention(384)
        self.down8_4 = self.Downsample(384, 384)
        self.enc4 = ResBlock(384, cond_dim=style_dim)
        self.attn4 = SelfAttention(384)
        self.down4_2 = self.Downsample(384, 384)
        self.enc2 = ResBlock(384, cond_dim=style_dim)
        self.attn2 = SelfAttention(384)
        self.up2_4 = self.Upsample(384, 384)
        self.merge4 = nn.Conv2d(384 * 2, 384, kernel_size=1)
        self.dec4 = ResBlock(384, cond_dim=style_dim)
        self.up4_8 = self.Upsample(384, 384)
        self.merge8 = nn.Conv2d(384 * 2, 384, kernel_size=1)
        self.dec8 = ResBlock(384, cond_dim=style_dim)
        self.up8_16 = self.Upsample(384, 256)
        self.merge16 = nn.Conv2d(256 * 2, 256, kernel_size=1)
        self.dec16 = ResBlock(256, cond_dim=style_dim)
        self.up16_32 = self.Upsample(256, 128)
        self.merge32 = nn.Conv2d(128 * 2, 128, kernel_size=1)
        self.dec32 = ResBlock(128, cond_dim=style_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, vocab_size - 1, kernel_size=1)
        )

    def _time_embedding(self, t: torch.Tensor):
        """Embed time step t into a vector."""
        t = t.view(-1).float()
        half_dim = (self.embed_dim + 1) // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)[:, :self.embed_dim]
        return self.time_mlp(emb)

    def forward(
        self,
        noisy_target: torch.Tensor,
        source_content: torch.Tensor,
        pack_id: torch.Tensor,
        t: torch.Tensor,
    ):
        # Extract structure from source exemplar
        content_features = self.content_encoder(source_content)  # [B, style_dim, H, W]

        # Embed the noisy input tokens
        x = self.token_embedding(noisy_target.long())  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Concatenate content structure with noisy input
        x = torch.cat([x, content_features], dim=1)  # [B, embed_dim + style_dim, H, W]

        # Pack style (aesthetic knowledge)
        pack_vec = self.pack_embedding(pack_id.long())  # [B, style_dim]

        # Time conditioning
        time_vec = self._time_embedding(t)  # [B, style_dim]

        # Combined conditioning for FiLM
        style_cond = time_vec + pack_vec  # [B, style_dim]

        # UNet with FiLM conditioning
        x = self.in_proj(x)
        skip32 = self.enc32(x, style_cond)
        x = self.down32_16(skip32)
        skip16 = self.enc16(x, style_cond)
        x = self.down16_8(skip16)
        skip8 = self.enc8(x, style_cond)
        skip8 = self.attn8(skip8)
        x = self.down8_4(skip8)
        skip4 = self.enc4(x, style_cond)
        skip4 = self.attn4(skip4)
        x = self.down4_2(skip4)
        center = self.enc2(x, style_cond)
        center = self.attn2(center)

        x = self.up2_4(center)
        x = self.merge4(torch.cat([x, skip4], dim=1))
        x = self.dec4(x, style_cond)
        x = self.up4_8(x)
        x = self.merge8(torch.cat([x, skip8], dim=1))
        x = self.dec8(x, style_cond)
        x = self.up8_16(x)
        x = self.merge16(torch.cat([x, skip16], dim=1))
        x = self.dec16(x, style_cond)
        x = self.up16_32(x)
        x = self.merge32(torch.cat([x, skip32], dim=1))
        x = self.dec32(x, style_cond)
        logits = self.out(x)
        return logits
