from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        groups = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class _ASPP(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
                nn.Conv2d(channels, channels, 3, padding=4, dilation=4, bias=False),
                nn.Conv2d(channels, channels, 3, padding=6, dilation=6, bias=False),
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([branch(x) for branch in self.branches], dim=1))


class _AttentionMerge(nn.Module):
    def __init__(self, dec_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        inter = max(out_ch // 2, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(dec_ch + skip_ch, inter, 1, bias=False),
            nn.GroupNorm(1, inter),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter, 1, 1),
            nn.Sigmoid(),
        )
        self.block = _ConvGNAct(dec_ch + skip_ch, out_ch)

    def forward(self, dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if dec.shape[-2:] != skip.shape[-2:]:
            dec = F.interpolate(dec, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        weight = self.gate(torch.cat([dec, skip], dim=1))
        return self.block(torch.cat([dec, skip * weight], dim=1))


class _LowDicePrior(nn.Module):
    """Image prior for small, unilateral and boundary-complex lesions."""

    def __init__(self):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # When a prior probability channel is appended, handcrafted CXR priors
        # should still be computed from the raw radiograph channel only.
        gray = x[:, :1]
        low = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        high = (gray - low).abs()
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()

        b, _, h, w = gray.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        yy = yy.expand(b, 1, h, w)
        xx = xx.expand(b, 1, h, w)
        return torch.cat([gray, low, high, edge, lap, xx, yy], dim=1)


class LowDiceRefineNet2D(nn.Module):
    """Specialist segmentation net for v18 low-Dice QaTa samples.

    It keeps a high-resolution path and adds handcrafted edge/contrast/coordinate
    priors, because the low-Dice analysis shows that failures are dominated by
    small unilateral lesions, boundary-complex masks and large centroid shifts.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        use_boundary_head: bool = True,
    ):
        super().__init__()
        self.use_boundary_head = use_boundary_head
        self.prior = _LowDicePrior()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stem = _ConvGNAct(in_channels + 7, c1)
        self.enc1 = nn.Sequential(_ConvGNAct(c1, c1), _SEBlock(c1))
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False)
        self.enc2 = nn.Sequential(_ConvGNAct(c2, c2), _SEBlock(c2))
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False)
        self.enc3 = nn.Sequential(_ConvGNAct(c3, c3), _SEBlock(c3))
        self.down3 = nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False)

        self.bottleneck = nn.Sequential(_ConvGNAct(c4, c4), _ASPP(c4), _SEBlock(c4))

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _AttentionMerge(c3, c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _AttentionMerge(c2, c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _AttentionMerge(c1, c1, c1)

        self.local_refine = nn.Sequential(
            _ConvGNAct(c1 + 7, c1),
            _SEBlock(c1),
        )
        self.seg_head = nn.Conv2d(c1, out_channels, 1)
        self.aux_head = nn.Conv2d(c2, out_channels, 1)
        if self.use_boundary_head:
            self.boundary_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor):
        prior = self.prior(x)
        x0 = self.stem(torch.cat([x, prior], dim=1))
        s1 = self.enc1(x0)
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        b = self.bottleneck(self.down3(s3))

        d3 = self.dec3(self.up3(b), s3)
        d2 = self.dec2(self.up2(d3), s2)
        d1 = self.dec1(self.up1(d2), s1)
        d1 = self.local_refine(torch.cat([d1, prior], dim=1))
        seg = self.seg_head(d1)

        outputs = {"seg": seg}
        if self.training:
            outputs["ds2"] = F.interpolate(
                self.aux_head(d2),
                size=seg.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if self.use_boundary_head:
            outputs["boundary"] = self.boundary_head(d1)
        return outputs


def build_lowdice_refinenet2d(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
):
    return LowDiceRefineNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=16,
        use_boundary_head=use_boundary_head,
    )


def build_lowdice_prior_refinenet2d(
    in_channels: int = 2,
    out_channels: int = 1,
    use_boundary_head: bool = True,
):
    return LowDiceRefineNet2D(
        in_channels=2,
        out_channels=out_channels,
        base_channels=16,
        use_boundary_head=use_boundary_head,
    )


if __name__ == "__main__":
    model = build_lowdice_refinenet2d()
    x = torch.randn(2, 1, 224, 224)
    model.train()
    out = model(x)
    print({k: v.shape for k, v in out.items()})
    model.eval()
    print(model(x)["seg"].shape)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
