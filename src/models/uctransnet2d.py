from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ChannelTokenFusion(nn.Module):
    """
    Fuse multi-scale features in a channel-token space (simplified UCTransNet idea).
    """
    def __init__(self, in_channels_list: List[int], embed_dim: int = 128, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, 1, bias=False) for c in in_channels_list])

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.to_gate = nn.ModuleList([nn.Linear(embed_dim, c) for c in in_channels_list])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats: list of skip features at multiple scales
        tokens = []
        for f, p in zip(feats, self.proj):
            z = p(f)                      # (B, E, H, W)
            z = z.mean(dim=(2, 3))        # (B, E) global average pooling as channel token
            tokens.append(z)
        t = torch.stack(tokens, dim=1)   # (B, S, E), S=#scales
        t = self.norm(t)
        t = self.tr(t)                   # (B, S, E)

        gated = []
        for i, f in enumerate(feats):
            gate = torch.sigmoid(self.to_gate[i](t[:, i, :]))  # (B, C_i)
            gate = gate.unsqueeze(-1).unsqueeze(-1)            # (B, C_i, 1, 1)
            gated.append(f * gate)
        return gated


class UCTransNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = _ConvBlock(in_channels, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.enc4 = _ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.fuse = _ChannelTokenFusion([c1, c2, c3], embed_dim=128, num_layers=2, num_heads=4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec3 = _ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = _ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = _ConvBlock(c1 + c1, c1)

        self.out = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        x = self.pool(s1)

        s2 = self.enc2(x)
        x = self.pool(s2)

        s3 = self.enc3(x)
        x = self.pool(s3)

        x = self.enc4(x)

        # UCTransNet-like: fuse skip connections before decoding
        s1g, s2g, s3g = self.fuse([s1, s2, s3])

        x = self.up3(x)
        x = torch.cat([x, s3g], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, s2g], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, s1g], dim=1)
        x = self.dec1(x)

        return self.out(x)


def build_uctransnet2d(in_channels: int = 1, out_channels: int = 1):
    return UCTransNet2D(in_channels=in_channels, out_channels=out_channels, base=32)
