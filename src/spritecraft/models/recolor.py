"""Lightweight RecolorNet for structure-preserving color transfer.

This model is trained to recolor a vanilla texture to match the style
of a target resource pack while preserving the exact vanilla structure.
It serves as a fallback for complex textures where the diffusion model
produces structurally incoherent outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spritecraft.models.unet import ResBlock, timestep_embedding


class RecolorNet(nn.Module):
    """Small U-Net for structure-preserving color transfer.

    Takes a vanilla content texture and style reference textures,
    produces a recolored output that matches the target pack's color
    palette while preserving the vanilla pixel structure.
    """

    def __init__(
        self,
        in_channels: int = 3,
        style_channels: int = 64,
        base_channels: int = 64,
        num_style_refs: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.style_channels = style_channels
        self.base_channels = base_channels
        self.num_style_refs = num_style_refs

        self.style_encoder = nn.Sequential(
            nn.Conv2d(in_channels, style_channels, 3, padding=1),
            ResBlock(style_channels),
            nn.Conv2d(style_channels, style_channels, 4, 2, 1),
            ResBlock(style_channels),
            nn.Conv2d(style_channels, style_channels, 4, 2, 1),
            ResBlock(style_channels),
        )

        self.content_enc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, 4, 2, 1),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, 4, 2, 1),
            ResBlock(base_channels),
        )

        self.style_proj = nn.Sequential(
            nn.Conv2d(style_channels, base_channels, 1),
            nn.SiLU(),
        )

        self.bottleneck = nn.Sequential(
            ResBlock(base_channels),
            ResBlock(base_channels),
        )

        self.dec1 = nn.Sequential(
            ResBlock(base_channels),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1),
        )
        self.dec2 = nn.Sequential(
            ResBlock(base_channels // 2),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 2, 4, 2, 1),
        )
        self.dec_merge = ResBlock(base_channels // 2 + in_channels)
        self.out_conv = nn.Conv2d(base_channels // 2 + in_channels, in_channels, 1)

        self._init_weights()
        total = sum(p.numel() for p in self.parameters())
        print(f"RecolorNet initialized with {total / 1e6:.2f}M parameters")

    def _init_weights(self):
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        content_rgb: torch.Tensor,
        style_refs: torch.Tensor,
        style_ref_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _, H, W = content_rgb.shape
        N = style_refs.shape[1]

        style_refs_flat = style_refs.reshape(B * N, self.in_channels, H, W)
        style_feat = self.style_encoder(style_refs_flat)
        style_feat = style_feat.view(B, N, self.style_channels, 8, 8)

        if style_ref_mask is not None:
            mask = style_ref_mask.view(B, N, 1, 1, 1).float()
        else:
            mask = torch.ones(B, N, 1, 1, 1, device=style_feat.device)
        style_feat = (style_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        style_cond = self.style_proj(style_feat)

        content_skip = content_rgb
        content_feat = self.content_enc(content_rgb)

        x = content_feat + style_cond
        x = self.bottleneck(x)

        x = self.dec1(x)
        x = self.dec2(x)

        x = torch.cat([x, F.interpolate(content_skip, size=x.shape[2:], mode="bilinear", align_corners=False)], dim=1)
        x = self.dec_merge(x)
        x = self.out_conv(x)

        return (content_skip + x).clamp(0.0, 1.0)


class RecolorLoss(nn.Module):
    """Combined loss for training RecolorNet."""

    def __init__(
        self,
        recon_weight: float = 1.0,
        edge_weight: float = 0.5,
        color_weight: float = 1.5,
        content_identity_weight: float = 0.1,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight
        self.content_identity_weight = content_identity_weight

    def _edge_map(self, rgb: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        return torch.cat([
            F.pad(gx, (0, 1, 0, 0)),
            F.pad(gy, (0, 0, 0, 1)),
        ], dim=1)

    def _channel_stats(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = rgb.shape[0]
        flat = rgb.reshape(B, 3, -1)
        return flat.mean(dim=2), flat.std(dim=2, unbiased=False).clamp_min(1e-4)

    def forward(
        self,
        pred: torch.Tensor,
        content: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        recon_loss = F.l1_loss(pred, target)

        pred_edges = self._edge_map(pred)
        content_edges = self._edge_map(content)
        edge_loss = F.l1_loss(pred_edges, content_edges)

        pred_mean, pred_std = self._channel_stats(pred)
        target_mean, target_std = self._channel_stats(target)
        color_loss = F.l1_loss(pred_mean, target_mean) + 0.5 * F.l1_loss(pred_std, target_std)

        content_identity = F.l1_loss(pred, content)

        total = (
            self.recon_weight * recon_loss
            + self.edge_weight * edge_loss
            + self.color_weight * color_loss
            + self.content_identity_weight * content_identity
        )
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "edge_loss": edge_loss,
            "color_loss": color_loss,
            "content_identity_loss": content_identity,
        }
