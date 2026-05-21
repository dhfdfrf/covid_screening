from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transunet2d_v11 import (
    _BoundaryRefinement,
    _ConvBlock,
    _SpatialGate,
    _WindowedTransformerEncoder,
    tta_inference,
)


class _FrequencyPriorExtractor(nn.Module):
    """Extracts a soft lesion prior from low/high-frequency CXR cues."""

    def __init__(self, channels: int = 16):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        )
        sobel_y = sobel_x.t()
        laplace = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32,
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("laplace", laplace.view(1, 1, 3, 3))
        self.encoder = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gray = image.mean(dim=1, keepdim=True)
        low = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        high = gray - low
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()

        feat = self.encoder(torch.cat([gray, low, high.abs(), edge, lap], dim=1))
        return feat, self.head(feat)


class _FrequencyResidualAdapter(nn.Module):
    """Zero-initialized residual adapter for scale-specific prior injection."""

    def __init__(self, feat_ch: int, prior_ch: int):
        super().__init__()
        hidden = max(feat_ch // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch + prior_ch + 1, hidden, 1, bias=False),
            nn.InstanceNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, feat_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        feat: torch.Tensor,
        prior_feat: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        prior_feat = F.interpolate(
            prior_feat,
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prior_prob = F.interpolate(
            torch.sigmoid(prior_logits),
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        delta = self.net(torch.cat([feat, prior_feat, prior_prob], dim=1))
        return feat + torch.tanh(self.gamma) * delta


class BoundaryAwareTransUNet2D_v18(nn.Module):
    """v18: stable v12/v14 TransUNet with frequency-prior residual adapters."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        use_boundary_head: bool = True,
        use_deep_supervision: bool = True,
        window_size: int = 4,
        prior_channels: int = 16,
    ):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.use_boundary_head = use_boundary_head
        self.use_deep_supervision = use_deep_supervision

        self.prior_branch = _FrequencyPriorExtractor(channels=prior_channels)

        self.enc1 = _ConvBlock(in_channels, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.enc4 = _ConvBlock(c3, c4)
        self.down = nn.MaxPool2d(2)

        self.adapter1 = _FrequencyResidualAdapter(c1, prior_channels)
        self.adapter2 = _FrequencyResidualAdapter(c2, prior_channels)
        self.adapter3 = _FrequencyResidualAdapter(c3, prior_channels)
        self.adapter4 = _FrequencyResidualAdapter(c4, prior_channels)

        self.transformer = _WindowedTransformerEncoder(
            d=c4,
            nhead=num_heads,
            nlayers=num_transformer_layers,
            drop=dropout,
            drop_path=drop_path,
            ws=window_size,
        )

        self.up4 = nn.ConvTranspose2d(c4, c4, 2, stride=2)
        self.dec4 = _ConvBlock(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _ConvBlock(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _ConvBlock(c1 + c1, c1)

        self.gate4 = _SpatialGate(c4, c4, max(c4 // 2, 1))
        self.gate3 = _SpatialGate(c3, c3, max(c3 // 2, 1))
        self.gate2 = _SpatialGate(c2, c2, max(c2 // 2, 1))
        self.gate1 = _SpatialGate(c1, c1, max(c1 // 2, 1))

        self.coarse_out = nn.Conv2d(c1, out_channels, 1)
        self.refine = _BoundaryRefinement(c1, out_channels)

        if self.use_deep_supervision:
            self.ds4 = nn.Conv2d(c4, out_channels, 1)
            self.ds3 = nn.Conv2d(c3, out_channels, 1)
            self.ds2 = nn.Conv2d(c2, out_channels, 1)

        if self.use_boundary_head:
            self.boundary_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor):
        orig_h, orig_w = x.shape[2], x.shape[3]
        prior_feat, prior_logits = self.prior_branch(x)

        s1 = self.adapter1(self.enc1(x), prior_feat, prior_logits)
        x_down = self.down(s1)
        s2 = self.adapter2(self.enc2(x_down), prior_feat, prior_logits)
        x_down = self.down(s2)
        s3 = self.adapter3(self.enc3(x_down), prior_feat, prior_logits)
        x_down = self.down(s3)
        s4 = self.adapter4(self.enc4(x_down), prior_feat, prior_logits)
        x_down = self.down(s4)

        b, c, h, w = x_down.shape
        tokens = x_down.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens, h, w)
        x_dec = tokens.transpose(1, 2).reshape(b, c, h, w)

        x_dec = self.up4(x_dec)
        x_dec = torch.cat([x_dec, self.gate4(x_dec, s4)], dim=1)
        d4 = self.dec4(x_dec)

        x_dec = self.up3(d4)
        x_dec = torch.cat([x_dec, self.gate3(x_dec, s3)], dim=1)
        d3 = self.dec3(x_dec)

        x_dec = self.up2(d3)
        x_dec = torch.cat([x_dec, self.gate2(x_dec, s2)], dim=1)
        d2 = self.dec2(x_dec)

        x_dec = self.up1(d2)
        x_dec = torch.cat([x_dec, self.gate1(x_dec, s1)], dim=1)
        d1 = self.dec1(x_dec)

        coarse = self.coarse_out(d1)
        seg_logits = coarse + self.refine(d1, coarse)

        outputs = {"seg": seg_logits, "prior": prior_logits}
        if self.use_deep_supervision and self.training:
            outputs["ds4"] = F.interpolate(
                self.ds4(d4),
                (orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            outputs["ds3"] = F.interpolate(
                self.ds3(d3),
                (orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            outputs["ds2"] = F.interpolate(
                self.ds2(d2),
                (orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        if self.use_boundary_head:
            outputs["boundary"] = self.boundary_head(d1)
        return outputs


def build_transunet2d_v18(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
    use_deep_supervision: bool = True,
):
    return BoundaryAwareTransUNet2D_v18(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=32,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.0,
        drop_path=0.1,
        use_boundary_head=use_boundary_head,
        use_deep_supervision=use_deep_supervision,
        window_size=4,
        prior_channels=16,
    )


if __name__ == "__main__":
    model = build_transunet2d_v18()
    x = torch.randn(2, 1, 224, 224)
    model.train()
    out = model(x)
    print("[Train] seg:", out["seg"].shape)
    print("[Train] prior:", out["prior"].shape)
    print("[Train] aux:", sorted(out.keys()))
    model.eval()
    print("[Eval] seg:", model(x)["seg"].shape)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
