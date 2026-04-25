from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR
except Exception:
    SwinUNETR = None


class _FallbackSmallUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 24, return_features: bool = False):
        super().__init__()
        self.return_features = return_features
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(base * 4, base * 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor):
        s1 = self.enc1(x)
        x = self.pool(s1)
        s2 = self.enc2(x)
        x = self.pool(s2)
        x = self.enc3(x)
        bottleneck = x
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)
        logits = self.out(x)
        if self.return_features:
            return logits, bottleneck
        return logits


class SwinUNet2DWrapper(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, image_size: Optional[Tuple[int, int]] = None, return_features: bool = False):
        super().__init__()
        self.return_features = return_features
        self.image_size = image_size or (224, 224)
        if SwinUNETR is None:
            self.backbone = _FallbackSmallUNet(in_channels, out_channels, return_features=return_features)
            self.uses_fallback = True
        else:
            self.backbone = SwinUNETR(
                img_size=self.image_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=24,
                use_checkpoint=True,
                spatial_dims=2,
            )
            self.uses_fallback = False
            self.feature_proj = nn.Sequential(
                nn.Conv2d(out_channels, 24, 1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor):
        if self.uses_fallback:
            return self.backbone(x)
        logits = self.backbone(x)
        if self.return_features:
            feat = self.feature_proj(torch.sigmoid(logits))
            return logits, feat
        return logits


def build_swin_unet2d(in_channels: int = 1, out_channels: int = 1, image_size: Optional[Tuple[int, int]] = None, return_features: bool = False):
    return SwinUNet2DWrapper(in_channels=in_channels, out_channels=out_channels, image_size=image_size, return_features=return_features)
