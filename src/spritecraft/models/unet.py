"""U-Net model architecture."""

import math

import torch
import torch.nn as nn


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


class SupportAggregator(nn.Module):
    """Aggregate per-support style pairs with content-conditioned attention."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        content_map: torch.Tensor,
        support_pair_maps: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_supports, channels, height, width = support_pair_maps.shape
        content_summary = content_map.mean(dim=(2, 3))
        support_summary = support_pair_maps.mean(dim=(3, 4))

        query = self.query_proj(self.norm(content_summary)).unsqueeze(1)
        key = self.key_proj(self.norm(support_summary))
        value = self.value_proj(support_summary)
        attn_scores = torch.matmul(query, key.transpose(1, 2)).squeeze(1) / math.sqrt(channels)

        if support_mask.dtype != torch.bool:
            support_mask = support_mask.to(dtype=torch.bool)
        attn_scores = attn_scores.masked_fill(~support_mask, float("-inf"))

        empty_rows = ~support_mask.any(dim=1)
        if empty_rows.any():
            attn_scores = attn_scores.masked_fill(empty_rows[:, None], 0.0)

        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = attn_weights * support_mask.to(dtype=attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        attended_summary = torch.sum(attn_weights.unsqueeze(-1) * value, dim=1)
        attended_summary = self.output_proj(attended_summary)
        attended_map = torch.sum(
            support_pair_maps * attn_weights[:, :, None, None, None],
            dim=1,
        )
        return attended_map, attended_summary


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
        self.support_pair_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        self.support_aggregator = SupportAggregator(embed_dim)
        self.support_cond_proj = nn.Linear(embed_dim, embed_dim)
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

    def _embed_support_pairs(
        self,
        support_content_ref: torch.Tensor,
        support_style_ref: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed and aggregate support exemplars with content-conditioned attention."""
        if support_content_ref.dim() == 3:
            support_content_ref = support_content_ref.unsqueeze(1)
            support_style_ref = support_style_ref.unsqueeze(1)
        if support_mask is None:
            support_mask = torch.ones(
                support_content_ref.shape[:2],
                device=support_content_ref.device,
                dtype=torch.bool,
            )
        elif support_mask.dim() == 1:
            support_mask = support_mask.unsqueeze(0)

        support_content = self.token_embedding(support_content_ref.long()).permute(0, 1, 4, 2, 3)
        support_style = self.token_embedding(support_style_ref.long()).permute(0, 1, 4, 2, 3)

        batch_size, num_supports, _channels, height, width = support_content.shape
        support_pairs = torch.cat([support_content, support_style], dim=2).reshape(
            batch_size * num_supports,
            self.embed_dim * 2,
            height,
            width,
        )
        support_pairs = self.support_pair_proj(support_pairs)
        return support_pairs.reshape(batch_size, num_supports, self.embed_dim, height, width), support_mask
    
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
    
    def forward(
        self,
        noisy_target: torch.Tensor,
        content_ref: torch.Tensor,
        support_content_ref: torch.Tensor,
        support_style_ref: torch.Tensor,
        support_mask: torch.Tensor | None,
        t: torch.Tensor,
    ):
        target = self._embed_tokens(noisy_target)
        content = self._embed_tokens(content_ref)
        support_pairs, support_mask = self._embed_support_pairs(
            support_content_ref,
            support_style_ref,
            support_mask=support_mask,
        )
        support_map, support_summary = self.support_aggregator(content, support_pairs, support_mask)

        cond = self._time_embedding(t) + self.support_cond_proj(support_summary)
        x = torch.cat([target, content, support_map], dim=1)
        x = self.in_proj(x)

        skip32 = self.enc32(x, cond)
        x = self.down32_16(skip32)
        skip16 = self.enc16(x, cond)
        x = self.down16_8(skip16)
        skip8 = self.enc8(x, cond)
        skip8 = self.attn8(skip8)
        x = self.down8_4(skip8)
        center = self.enc4(x, cond)
        x = self.attn4(center)

        x = self.up4_8(x)
        x = self.merge8(torch.cat([x, skip8], dim=1))
        x = self.dec8(x, cond)
        x = self.up8_16(x)
        x = self.merge16(torch.cat([x, skip16], dim=1))
        x = self.dec16(x, cond)
        x = self.up16_32(x)
        x = self.merge32(torch.cat([x, skip32], dim=1))
        x = self.dec32(x, cond)
        return self.out(x)
