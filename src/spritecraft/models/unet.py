"""Style-aware U-Net model for per-pack RGB diffusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.activation(self.norm1(x))
        h = self.conv1(h)
        h = self.activation(self.norm2(h))
        h = self.conv2(h)
        return h + residual


class CrossAttention(nn.Module):
    """Cross-attention module for content-to-style attention."""

    def __init__(self, dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, dim)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(context_dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(context_dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Initialize output projection to zero
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        _,CtxC, _, _ = context.shape

        # Normalize and project
        x_norm = self.norm(x)
        q = self.q_proj(x_norm)  # [B, C, H, W]
        k = self.k_proj(context)  # [B, C, H, W]
        v = self.v_proj(context)  # [B, C, H, W]

        # Reshape for multi-head attention
        # [B, heads, head_dim, H, W] -> [B, heads, HW, head_dim]
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(2, 3).reshape(B, C, H, W)
        out = self.out_proj(out)
        return x + out


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class StyleAwareUNet(nn.Module):
    """Small U-Net (~5M parameters) for 32x32 pixel-art style transfer.
    
    Uses explicit style reference images via cross-attention.
    Input/Output: RGB images in [0, 1] range.
    """

    def __init__(
        self,
        in_channels: int = 3,
        style_channels: int = 64,
        base_channels: int = 128,
        num_style_refs: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.style_channels = style_channels
        self.base_channels = base_channels
        self.num_style_refs = num_style_refs

        # Content encoder: processes vanilla target block
        self.content_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # 16x16
            ResBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1),  # 8x8
            ResBlock(base_channels * 2),
        )

        # Style encoder: processes reference textures from target pack
        self.style_encoder = nn.Sequential(
            nn.Conv2d(in_channels, style_channels, 3, padding=1),
            ResBlock(style_channels),
            nn.Conv2d(style_channels, style_channels, 4, 2, 1),  # 16x16
            ResBlock(style_channels),
            nn.Conv2d(style_channels, style_channels, 4, 2, 1),  # 8x8
            ResBlock(style_channels),
        )

        # Cross-attention: content queries attend to style keys/values
        self.cross_attn = CrossAttention(
            dim=base_channels * 2,
            context_dim=style_channels,
            num_heads=4,
        )

        # Time embedding (for diffusion)
        time_embed_dim = base_channels * 2
        self.time_embed = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, time_embed_dim),
        )

        # Decoder with skip connections
        self.dec1 = nn.Sequential(
            ResBlock(base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # 8x8 -> 16x16
        )
        self.dec2 = nn.Sequential(
            ResBlock(base_channels),
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1),  # 16x16 -> 32x32
        )
        self.dec3 = nn.Sequential(
            ResBlock(base_channels),
            nn.Conv2d(base_channels, in_channels, 1),  # Predict RGB
        )

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"StyleAwareUNet initialized with {total / 1e6:.2f}M parameters")

    def forward(
        self,
        noisy_target: torch.Tensor,
        content_source: torch.Tensor,
        style_refs: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            noisy_target: [B, 3, H, W] noisy target RGB
            content_source: [B, 3, H, W] vanilla content RGB
            style_refs: [B, N, 3, H, W] N reference textures from target pack
            t: [B] diffusion timesteps
        
        Returns:
            [B, 3, H, W] predicted denoised RGB
        """
        B = noisy_target.shape[0]

        # Encode content (vanilla structure)
        content_feat = self.content_encoder(content_source)  # [B, C*2, 8, 8]

        # Encode style references (average pool across N refs)
        N = style_refs.shape[1]
        style_refs = style_refs.view(B * N, self.in_channels, noisy_target.shape[2], noisy_target.shape[3])
        style_feat = self.style_encoder(style_refs)  # [B*N, style_C, 8, 8]
        style_feat = style_feat.view(B, N, self.style_channels, 8, 8)
        style_feat = style_feat.mean(dim=1)  # [B, style_C, 8, 8]

        # Cross-attention: content attends to style
        content_feat = self.cross_attn(content_feat, style_feat)

        # Add time embedding
        time_emb = self.time_embed(timestep_embedding(t, 64))
        content_feat = content_feat + time_emb[:, :, None, None]

        # Decode
        x = self.dec1(content_feat)  # [B, C, 16, 16]
        x = self.dec2(x)  # [B, C, 32, 32]
        x = self.dec3(x)  # [B, 3, 32, 32]
        return x



