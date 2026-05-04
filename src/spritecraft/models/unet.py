"""U-Net model architecture."""

import torch
import torch.nn as nn
import math


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int = 128):
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
        residual = x    # To be added back at the end
        scale1, shift1, scale2, shift2 = self._get_film_tr(cond)
        
        # First convolutional layer
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h = self.activation(h)
        h = self.conv1(h)
        
        # Second convolutional layer
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
        
        # Initialize projection to zero
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Quick aliases for readability
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)  # Each: [B, C, H, W]
        
        # Reshape to [B, heads, HW, dim_per_head]
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        
        # Calculate using FlashAttention
        h = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        h = h.transpose(2, 3).reshape(B, C, H, W)
        h = self.proj(h)
        return x + h


class UNet(nn.Module):
    """U-Net for masked diffusion on pixel-art textures."""

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

    def __init__(self, vocab_size: int = 257, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.in_proj = nn.Conv2d(embed_dim * 3, 96, kernel_size=3, padding=1)
        self.enc32 = ResBlock(96, cond_dim=embed_dim)
        self.down32_16 = self.Downsample(96, 192)
        self.enc16 = ResBlock(192, cond_dim=embed_dim)
        self.down16_8 = self.Downsample(192, 192)
        self.enc8 = ResBlock(192, cond_dim=embed_dim)
        self.attn8 = SelfAttention(192)
        self.down8_4 = self.Downsample(192, 192)
        self.enc4 = ResBlock(192, cond_dim=embed_dim)
        self.attn4 = SelfAttention(192)
        self.up4_8 = self.Upsample(192, 192)
        self.merge8 = nn.Conv2d(192 * 2, 192, kernel_size=1)
        self.dec8 = ResBlock(192, cond_dim=embed_dim)
        self.up8_16 = self.Upsample(192, 192)
        self.merge16 = nn.Conv2d(192 * 2, 192, kernel_size=1)
        self.dec16 = ResBlock(192, cond_dim=embed_dim)
        self.up16_32 = self.Upsample(192, 96)
        self.merge32 = nn.Conv2d(96 * 2, 96, kernel_size=1)
        self.dec32 = ResBlock(96, cond_dim=embed_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(32, 96),
            nn.SiLU(),
            nn.Conv2d(96, vocab_size - 1, kernel_size=1)
        )


    def _embed_tokens(self, x: torch.Tensor):
        """Embed input tokens and reshape to [B, C, H, W]."""
        x = self.token_embedding(x.long())
        return x.permute(0, 3, 1, 2)  # [B, C, H, W]
    
    def _time_embedding(self, t: torch.Tensor):
        """Embed time step t into a vector."""
        t = t.view(-1).float()
        half_dim = (self.token_embedding.embedding_dim + 1) // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)[:, : self.token_embedding.embedding_dim]
        return self.time_mlp(emb)
    
    def forward(self, noisy_target: torch.Tensor, content_ref: torch.Tensor, style_ref: torch.Tensor, t: torch.Tensor):
        # Preprocess image batches
        target = self._embed_tokens(noisy_target)
        content = self._embed_tokens(content_ref)
        style = self._embed_tokens(style_ref)

        # Calculate cond tensor
        cond = self._time_embedding(t)

        # Build forwarding network
        x = torch.cat([target, content, style], dim=1)  # [B, C*3, H, W]
        x = self.in_proj(x)
        
        # Encoding sequence
        skip32 = self.enc32(x, cond)
        x = self.down32_16(skip32)
        skip16 = self.enc16(x, cond)
        x = self.down16_8(skip16)
        skip8 = self.enc8(x, cond)
        skip8 = self.attn8(skip8)
        x = self.down8_4(skip8)
        center = self.enc4(x, cond)
        x = self.attn4(center)

        # Decoding sequence
        x = self.up4_8(x)
        x = self.merge8(
            torch.cat([x, skip8], dim=1)
        )
        x = self.dec8(x, cond)
        x = self.up8_16(x)
        x = self.merge16(
            torch.cat([x, skip16], dim=1)
        )
        x = self.dec16(x, cond)
        x = self.up16_32(x)
        x = self.merge32(
            torch.cat([x, skip32], dim=1)
        )
        x = self.dec32(x, cond)
        return self.out(x)
