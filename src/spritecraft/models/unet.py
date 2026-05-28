"""Style-aware U-Net model for per-pack RGB diffusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_norm_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""

    def __init__(self, channels: int):
        super().__init__()
        num_groups = _resolve_group_norm_groups(channels)
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
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

        self.norm = nn.GroupNorm(_resolve_group_norm_groups(dim), dim)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(context_dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(context_dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Initialize output projection to zero
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        _, _ctx_c, ctx_h, ctx_w = context.shape

        # Normalize and project
        x_norm = self.norm(x)
        q = self.q_proj(x_norm)  # [B, C, H, W]
        k = self.k_proj(context)  # [B, C, H, W]
        v = self.v_proj(context)  # [B, C, H, W]

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, ctx_h * ctx_w).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, ctx_h * ctx_w).transpose(2, 3)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if context_mask is not None:
            mask = context_mask[:, None, None, :].to(dtype=torch.bool, device=attn.device)
            attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
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

        # Encode the current noisy sample and the vanilla source separately,
        # then fuse them early so the denoiser can use both structure and state.
        self.noisy_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.content_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc32 = ResBlock(base_channels)
        self.down16 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
        self.enc16 = ResBlock(base_channels * 2)
        self.down8 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.enc8 = ResBlock(base_channels * 2)

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

        # Decoder with explicit spatial skips to preserve high-contrast edges.
        self.dec1 = nn.Sequential(
            ResBlock(base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # 8x8 -> 16x16
        )
        self.skip16_proj = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        self.dec2 = nn.Sequential(
            ResBlock(base_channels),
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1),  # 16x16 -> 32x32
        )
        self.merge32 = ResBlock(base_channels)
        self.dec3 = nn.Sequential(
            ResBlock(base_channels),
            nn.Conv2d(base_channels, in_channels, 1),  # Predict diffusion noise
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
        style_ref_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            noisy_target: [B, 3, H, W] noisy target RGB
            content_source: [B, 3, H, W] vanilla content RGB
            style_refs: [B, N, 3, H, W] N reference textures from target pack
            t: [B] diffusion timesteps
            style_ref_mask: [B, N] boolean mask for valid references
        
        Returns:
            [B, 3, H, W] predicted denoised RGB
        """
        B, _, H, W = noisy_target.shape

        noisy_feat = self.noisy_in(noisy_target)
        content_feat = self.content_in(content_source)
        x32 = self.enc32(noisy_feat + content_feat)
        x16 = self.enc16(self.down16(x32))
        x8 = self.enc8(self.down8(x16))

        # Encode style references and preserve all support tokens rather than
        # collapsing them into an average feature map.
        N = style_refs.shape[1]
        style_refs_flat = style_refs.reshape(B * N, self.in_channels, H, W)
        style_feat = self.style_encoder(style_refs_flat)  # [B*N, style_C, 8, 8]
        style_feat = style_feat.view(B, N, self.style_channels, 8, 8)
        style_context = style_feat.permute(0, 2, 3, 1, 4).reshape(B, self.style_channels, 8, 8 * N)

        context_mask = None
        if style_ref_mask is not None:
            context_mask = style_ref_mask[:, :, None].expand(B, N, 64).reshape(B, N * 64)

        x8 = self.cross_attn(x8, style_context, context_mask=context_mask)

        # Add time embedding
        time_emb = self.time_embed(timestep_embedding(t, 64))
        x8 = x8 + time_emb[:, :, None, None]

        # Decode
        x = self.dec1(x8)  # [B, C, 16, 16]
        x = x + self.skip16_proj(x16)
        x = self.dec2(x)  # [B, C, 32, 32]
        x = self.merge32(x + x32)
        x = self.dec3(x)  # [B, 3, 32, 32]
        return x


