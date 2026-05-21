from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transunet2d_v11 import (
    _ConvBlock,
    _SpatialGate,
    _WindowedTransformerEncoder,
    tta_inference,
)


class _DilatedContext(nn.Module):
    """Lightweight ASPP-style context block, initialized as a residual branch."""

    def __init__(self, channels: int):
        super().__init__()
        branch_channels = max(channels // 4, 1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.InstanceNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                ),
                self._branch(channels, branch_channels, dilation=1),
                self._branch(channels, branch_channels, dilation=2),
                self._branch(channels, branch_channels, dilation=3),
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 4, channels, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Starts close to v11/v12 behavior, then learns extra context if useful.
        self.gamma = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _branch(in_channels: int, out_channels: int, dilation: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.gamma * self.fuse(context)


class _DualBoundaryRefinement(nn.Module):
    """Refines coarse logits with both prediction edges and image edges."""

    def __init__(self, feat_ch: int, out_ch: int):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        )
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(feat_ch + 4, feat_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(feat_ch, out_ch, 1)

    def _sobel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.conv2d(x, self.sobel_x, padding=1),
            F.conv2d(x, self.sobel_y, padding=1),
        )

    def forward(
        self,
        feat: torch.Tensor,
        coarse_seg: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        seg_prob = torch.sigmoid(coarse_seg)
        seg_ex, seg_ey = self._sobel(seg_prob)
        img = F.interpolate(
            image,
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        img_ex, img_ey = self._sobel(img)
        refined = self.edge_conv(torch.cat([feat, seg_ex, seg_ey, img_ex, img_ey], dim=1))
        return self.out_conv(refined)


class BoundaryAwareTransUNet2D_v13(nn.Module):
    """v13: v11/v12 backbone plus lightweight context and dual-edge refinement."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        drop_path: float = 0.08,
        use_boundary_head: bool = True,
        use_deep_supervision: bool = True,
        window_size: int = 4,
        decoder_dropout: float = 0.03,
    ):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.use_boundary_head = use_boundary_head
        self.use_deep_supervision = use_deep_supervision
        self.decoder_dropout = nn.Dropout2d(decoder_dropout) if decoder_dropout > 0 else nn.Identity()

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
        self.context = _DilatedContext(c4)

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
        self.refine = _DualBoundaryRefinement(c1, out_channels)

        if self.use_deep_supervision:
            self.ds4 = nn.Conv2d(c4, out_channels, 1)
            self.ds3 = nn.Conv2d(c3, out_channels, 1)
            self.ds2 = nn.Conv2d(c2, out_channels, 1)

        if self.use_boundary_head:
            self.boundary_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor):
        image = x
        orig_h, orig_w = x.shape[2], x.shape[3]

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
        x = self.context(x)

        x = self.up4(x)
        x = torch.cat([x, self.gate4(x, s4)], dim=1)
        d4 = self.decoder_dropout(self.dec4(x))

        x = self.up3(d4)
        x = torch.cat([x, self.gate3(x, s3)], dim=1)
        d3 = self.decoder_dropout(self.dec3(x))

        x = self.up2(d3)
        x = torch.cat([x, self.gate2(x, s2)], dim=1)
        d2 = self.decoder_dropout(self.dec2(x))

        x = self.up1(d2)
        x = torch.cat([x, self.gate1(x, s1)], dim=1)
        d1 = self.dec1(x)

        coarse = self.coarse_out(d1)
        seg_logits = coarse + self.refine(d1, coarse, image)

        outputs = {"seg": seg_logits}
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


def build_transunet2d_v13(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
    use_deep_supervision: bool = True,
):
    return BoundaryAwareTransUNet2D_v13(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=32,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.0,
        drop_path=0.08,
        use_boundary_head=use_boundary_head,
        use_deep_supervision=use_deep_supervision,
        window_size=4,
        decoder_dropout=0.03,
    )


if __name__ == "__main__":
    model = build_transunet2d_v13()
    x = torch.randn(2, 1, 224, 224)
    model.train()
    out = model(x)
    print("[Train] seg:", out["seg"].shape)
    print("[Train] aux:", sorted(out.keys()))
    model.eval()
    print("[Eval] seg:", model(x)["seg"].shape)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
