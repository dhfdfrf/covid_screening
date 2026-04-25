from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "in"):
        super().__init__()
        if norm == "bn":
            Norm = nn.BatchNorm2d
        elif norm == "gn":
            groups = min(32, out_ch)
            Norm = lambda c: nn.GroupNorm(groups, c)  # type: ignore
        else:
            Norm = nn.InstanceNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _posenc_2d_sincos(
    h: int,
    w: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create 2D sinusoidal positional encoding for (h*w) tokens.
    Returns: (1, h*w, dim)
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin-cos pos enc"

    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=dtype) / (dim // 4)
    omega = 1.0 / (10000 ** omega)

    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)

    pe = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    return pe.unsqueeze(0)  # (1, L, dim)


class _SpatialGate(nn.Module):
    """
    Lightweight attention gate for skip connections.
    gate: decoder feature after upsampling
    skip: encoder feature at the same spatial scale
    """
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(inter_ch),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attn = self.psi(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * attn


class BoundaryAwareTransUNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_boundary_head: bool = True,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        bottleneck_c = base_channels * 8

        self.use_boundary_head = use_boundary_head

        # Encoder
        self.enc1 = _ConvBlock(in_channels, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.enc4 = _ConvBlock(c3, c4)

        self.down = nn.MaxPool2d(2)

        # Transformer bottleneck
        enc_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_c,
            nhead=num_heads,
            dim_feedforward=bottleneck_c * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_transformer_layers
        )
        self.tr_norm = nn.LayerNorm(bottleneck_c)

        # Decoder
        self.up4 = nn.ConvTranspose2d(bottleneck_c, c4, kernel_size=2, stride=2)
        self.dec4 = _ConvBlock(c4 + c4, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(c1 + c1, c1)

        # Attention gates for skip connections
        self.gate4 = _SpatialGate(gate_ch=c4, skip_ch=c4, inter_ch=max(c4 // 2, 1))
        self.gate3 = _SpatialGate(gate_ch=c3, skip_ch=c3, inter_ch=max(c3 // 2, 1))
        self.gate2 = _SpatialGate(gate_ch=c2, skip_ch=c2, inter_ch=max(c2 // 2, 1))
        self.gate1 = _SpatialGate(gate_ch=c1, skip_ch=c1, inter_ch=max(c1 // 2, 1))

        # Main segmentation head
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

        # Auxiliary boundary head
        if self.use_boundary_head:
            self.boundary_head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Encoder
        s1 = self.enc1(x)          # (B, c1, H,   W)
        x = self.down(s1)          # (B, c1, H/2, W/2)

        s2 = self.enc2(x)          # (B, c2, H/2, W/2)
        x = self.down(s2)          # (B, c2, H/4, W/4)

        s3 = self.enc3(x)          # (B, c3, H/4, W/4)
        x = self.down(s3)          # (B, c3, H/8, W/8)

        s4 = self.enc4(x)          # (B, c4, H/8, W/8)
        x = self.down(s4)          # (B, c4, H/16, W/16)

        # Transformer bottleneck
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, L, C)
        tokens = tokens + _posenc_2d_sincos(h, w, c, tokens.device, tokens.dtype)
        tokens = self.tr_norm(tokens)
        tokens = self.transformer(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        # Decoder + gated skip fusion
        x = self.up4(x)            # -> H/8
        s4_att = self.gate4(x, s4)
        x = torch.cat([x, s4_att], dim=1)
        x = self.dec4(x)

        x = self.up3(x)            # -> H/4
        s3_att = self.gate3(x, s3)
        x = torch.cat([x, s3_att], dim=1)
        x = self.dec3(x)

        x = self.up2(x)            # -> H/2
        s2_att = self.gate2(x, s2)
        x = torch.cat([x, s2_att], dim=1)
        x = self.dec2(x)

        x = self.up1(x)            # -> H
        s1_att = self.gate1(x, s1)
        x = torch.cat([x, s1_att], dim=1)
        feat = self.dec1(x)

        seg_logits = self.out(feat)

        if self.use_boundary_head:
            boundary_logits = self.boundary_head(feat)
            return seg_logits, boundary_logits

        return seg_logits


def build_transunet2d(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
):
    return BoundaryAwareTransUNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=32,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.0,
        use_boundary_head=use_boundary_head,
    )