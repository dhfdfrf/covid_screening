from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transunet2d_v11 import (
    _BoundaryRefinement,
    _ConvBlock,
    _WindowedTransformerEncoder,
    tta_inference,
)


class _ImageLesionPrior(nn.Module):
    """Builds a lightweight lesion prior from intensity, local contrast, and edges."""

    def __init__(self, channels: int = 16):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        )
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.prior_head = nn.Conv2d(channels, 1, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.mean(dim=1, keepdim=True)
        gx = F.conv2d(image, self.sobel_x, padding=1)
        gy = F.conv2d(image, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        local_mean = F.avg_pool2d(image, kernel_size=7, stride=1, padding=3)
        local_contrast = (image - local_mean).abs()

        prior_features = self.stem(torch.cat([image, edge, local_mean, local_contrast], dim=1))
        prior_logits = self.prior_head(prior_features)
        return prior_features, prior_logits


class _PriorSpatialGate(nn.Module):
    """Attention gate modulated by an image-derived lesion prior.

    The extra prior modulation is controlled by a zero-initialized residual scale,
    so warm-starting from v12/v14 begins from the old behavior.
    """

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            nn.InstanceNorm2d(inter_ch),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            nn.InstanceNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.prior_proj = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        gate: torch.Tensor,
        skip: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        spatial_gate = self.psi(self.gate_proj(gate) + self.skip_proj(skip))
        prior = F.interpolate(
            torch.sigmoid(prior_logits),
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prior_gate = self.prior_proj(prior)
        prior_residual = 1.0 + torch.tanh(self.gamma) * (2.0 * prior_gate - 1.0)
        return skip * spatial_gate * prior_residual


class BoundaryAwareTransUNet2D_v16(nn.Module):
    """v16: stable v12 backbone with lesion-prior guided skip attention."""

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

        self.prior_branch = _ImageLesionPrior(channels=prior_channels)

        self.enc1 = _ConvBlock(in_channels, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.enc4 = _ConvBlock(c3, c4)
        self.down = nn.MaxPool2d(2)

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

        self.gate4 = _PriorSpatialGate(c4, c4, max(c4 // 2, 1))
        self.gate3 = _PriorSpatialGate(c3, c3, max(c3 // 2, 1))
        self.gate2 = _PriorSpatialGate(c2, c2, max(c2 // 2, 1))
        self.gate1 = _PriorSpatialGate(c1, c1, max(c1 // 2, 1))

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
        _, prior_logits = self.prior_branch(x)

        s1 = self.enc1(x)
        x = self.down(s1)
        s2 = self.enc2(x)
        x = self.down(s2)
        s3 = self.enc3(x)
        x = self.down(s3)
        s4 = self.enc4(x)
        x = self.down(s4)

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens, h, w)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        x = self.up4(x)
        x = torch.cat([x, self.gate4(x, s4, prior_logits)], dim=1)
        d4 = self.dec4(x)

        x = self.up3(d4)
        x = torch.cat([x, self.gate3(x, s3, prior_logits)], dim=1)
        d3 = self.dec3(x)

        x = self.up2(d3)
        x = torch.cat([x, self.gate2(x, s2, prior_logits)], dim=1)
        d2 = self.dec2(x)

        x = self.up1(d2)
        x = torch.cat([x, self.gate1(x, s1, prior_logits)], dim=1)
        d1 = self.dec1(x)

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


def build_transunet2d_v16(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
    use_deep_supervision: bool = True,
):
    return BoundaryAwareTransUNet2D_v16(
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
    model = build_transunet2d_v16()
    x = torch.randn(2, 1, 224, 224)
    model.train()
    out = model(x)
    print("[Train] seg:", out["seg"].shape)
    print("[Train] prior:", out["prior"].shape)
    print("[Train] aux:", sorted(out.keys()))
    model.eval()
    print("[Eval] seg:", model(x)["seg"].shape)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
