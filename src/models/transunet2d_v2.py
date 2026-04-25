from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Innovation 1: Multi-Scale Convolution Block (MSConv)
#   - Parallel branches with 1x1, 3x3, 5x5 (dilated 3x3) convolutions
#   - Channel shuffle for cross-branch information flow
# =============================================================================

class _ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)


class _MSConvBlock(nn.Module):
    """Multi-Scale Convolution Block with parallel receptive fields."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "in"):
        super().__init__()
        Norm = self._get_norm(norm)

        mid = out_ch // 3
        rem = out_ch - mid * 2  # handle indivisible channels

        # Branch 1: 1x1 (local details)
        self.br1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), Norm(mid), nn.ReLU(inplace=True),
        )
        # Branch 2: 3x3 (medium context)
        self.br3 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False), Norm(mid), nn.ReLU(inplace=True),
        )
        # Branch 3: 3x3 dilated r=2 (large context, replaces 5x5)
        self.br5 = nn.Sequential(
            nn.Conv2d(in_ch, rem, 3, padding=2, dilation=2, bias=False), Norm(rem), nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _get_norm(norm: str):
        if norm == "bn":
            return nn.BatchNorm2d
        elif norm == "gn":
            return lambda c: nn.GroupNorm(min(32, c), c)
        return nn.InstanceNorm2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.br1(x), self.br3(x), self.br5(x)], dim=1)
        return self.fuse(out)


# =============================================================================
# Innovation 2: Channel-Spatial Attention Gate (CSAG)
#   - SE-style channel attention + spatial attention for skip connections
#   - Richer gating than a single spatial sigmoid
# =============================================================================

class _CSAG(nn.Module):
    """Channel-Spatial Attention Gate for skip connections."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        # Spatial attention (similar to original but improved)
        self.gate_proj = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.skip_proj = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.spatial_attn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # Channel attention (SE-style on gated skip)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(skip_ch, skip_ch // 4),
            nn.ReLU(inplace=True),
            nn.Linear(skip_ch // 4, skip_ch),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Spatial gate
        s_attn = self.spatial_attn(self.gate_proj(gate) + self.skip_proj(skip))
        out = skip * s_attn

        # Channel gate
        c_attn = self.channel_attn(out).unsqueeze(-1).unsqueeze(-1)
        return out * c_attn


# =============================================================================
# Innovation 3: Windowed Transformer with Cross-Window Shift
#   - Reduces O(L^2) to O(L * win^2) complexity
#   - Alternating shifted windows for global connectivity (Swin-style)
# =============================================================================

def _posenc_2d_sincos(h, w, dim, device, dtype):
    assert dim % 4 == 0
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
    return pe.unsqueeze(0)


class _WindowedTransformerBlock(nn.Module):
    """Single transformer block with optional window shifting."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float, shift: bool = False, win_size: int = 4):
        super().__init__()
        self.shift = shift
        self.win_size = win_size
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def _window_partition(self, x: torch.Tensor, h: int, w: int):
        """x: (B, H*W, C) -> (B*nW, win*win, C)"""
        ws = self.win_size
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        if self.shift:
            x = torch.roll(x, shifts=(-ws // 2, -ws // 2), dims=(1, 2))
        # Pad if needed
        ph = (ws - h % ws) % ws
        pw = (ws - w % ws) % ws
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, 0, 0, pw, 0, ph))
        hp, wp = h + ph, w + pw
        x = x.view(b, hp // ws, ws, wp // ws, ws, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
        return x, hp, wp

    def _window_unpartition(self, x: torch.Tensor, hp: int, wp: int, h: int, w: int, b: int):
        ws = self.win_size
        c = x.shape[-1]
        x = x.view(b, hp // ws, wp // ws, ws, ws, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
        if self.shift:
            x = torch.roll(x, shifts=(ws // 2, ws // 2), dims=(1, 2))
        x = x[:, :h, :w, :].reshape(b, h * w, c)
        return x

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b = x.shape[0]
        residual = x
        x_norm = self.norm1(x)
        xw, hp, wp = self._window_partition(x_norm, h, w)
        xw, _ = self.attn(xw, xw, xw)
        x_norm = self._window_unpartition(xw, hp, wp, h, w, b)
        x = residual + x_norm

        x = x + self.ffn(self.norm2(x))
        return x


class _WindowedTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float, win_size: int = 4):
        super().__init__()
        ffn_dim = d_model * 4
        self.layers = nn.ModuleList([
            _WindowedTransformerBlock(
                d_model, nhead, ffn_dim, dropout,
                shift=(i % 2 == 1),  # alternate shift
                win_size=win_size,
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, h, w)
        return self.norm(x)


# =============================================================================
# Innovation 4: Deep Supervision
#   - Auxiliary segmentation outputs at multiple decoder scales
#   - Weighted combination during training for faster convergence
# =============================================================================

# (Integrated directly in the main model below)


# =============================================================================
# Main Model: BoundaryAwareTransUNet2D v2
# =============================================================================

class BoundaryAwareTransUNet2D_v2(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_boundary_head: bool = True,
        use_deep_supervision: bool = True,
        window_size: int = 4,
    ):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        bottleneck_c = c4

        self.use_boundary_head = use_boundary_head
        self.use_deep_supervision = use_deep_supervision

        # Encoder (MSConv blocks)
        self.enc1 = _MSConvBlock(in_channels, c1)
        self.enc2 = _MSConvBlock(c1, c2)
        self.enc3 = _MSConvBlock(c2, c3)
        self.enc4 = _MSConvBlock(c3, c4)
        self.down = nn.MaxPool2d(2)

        # Windowed Transformer bottleneck
        self.transformer = _WindowedTransformerEncoder(
            d_model=bottleneck_c,
            nhead=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            win_size=window_size,
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(bottleneck_c, c4, 2, stride=2)
        self.dec4 = _MSConvBlock(c4 + c4, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _MSConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _MSConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _MSConvBlock(c1 + c1, c1)

        # Channel-Spatial Attention Gates
        self.gate4 = _CSAG(c4, c4, max(c4 // 2, 1))
        self.gate3 = _CSAG(c3, c3, max(c3 // 2, 1))
        self.gate2 = _CSAG(c2, c2, max(c2 // 2, 1))
        self.gate1 = _CSAG(c1, c1, max(c1 // 2, 1))

        # Main output
        self.out = nn.Conv2d(c1, out_channels, 1)

        # Deep supervision heads
        if self.use_deep_supervision:
            self.ds4 = nn.Conv2d(c4, out_channels, 1)
            self.ds3 = nn.Conv2d(c3, out_channels, 1)
            self.ds2 = nn.Conv2d(c2, out_channels, 1)

        # Boundary head
        if self.use_boundary_head:
            self.boundary_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor):
        orig_h, orig_w = x.shape[2], x.shape[3]

        # ---------- Encoder ----------
        s1 = self.enc1(x);   x = self.down(s1)
        s2 = self.enc2(x);   x = self.down(s2)
        s3 = self.enc3(x);   x = self.down(s3)
        s4 = self.enc4(x);   x = self.down(s4)

        # ---------- Windowed Transformer ----------
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + _posenc_2d_sincos(h, w, c, tokens.device, tokens.dtype)
        tokens = self.transformer(tokens, h, w)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        # ---------- Decoder ----------
        x = self.up4(x)
        x = torch.cat([x, self.gate4(x, s4)], 1)
        d4 = self.dec4(x)

        x = self.up3(d4)
        x = torch.cat([x, self.gate3(x, s3)], 1)
        d3 = self.dec3(x)

        x = self.up2(d3)
        x = torch.cat([x, self.gate2(x, s2)], 1)
        d2 = self.dec2(x)

        x = self.up1(d2)
        x = torch.cat([x, self.gate1(x, s1)], 1)
        d1 = self.dec1(x)

        seg_logits = self.out(d1)

        outputs = {"seg": seg_logits}

        # Deep supervision (upsampled to original size)
        if self.use_deep_supervision and self.training:
            outputs["ds4"] = F.interpolate(self.ds4(d4), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            outputs["ds3"] = F.interpolate(self.ds3(d3), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            outputs["ds2"] = F.interpolate(self.ds2(d2), size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        if self.use_boundary_head:
            outputs["boundary"] = self.boundary_head(d1)

        return outputs


# =============================================================================
# Deep Supervision Loss Helper
# =============================================================================

class DeepSupervisionLoss(nn.Module):
    """Weighted combination of main + auxiliary losses."""

    def __init__(self, loss_fn: nn.Module | None = None, weights: tuple = (1.0, 0.4, 0.2, 0.1)):
        super().__init__()
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()
        self.weights = weights  # (main, ds2, ds3, ds4)

    def forward(self, outputs: dict, target: torch.Tensor) -> torch.Tensor:
        loss = self.weights[0] * self.loss_fn(outputs["seg"], target)
        for i, key in enumerate(["ds2", "ds3", "ds4"], 1):
            if key in outputs:
                loss += self.weights[i] * self.loss_fn(outputs[key], target)
        return loss


# =============================================================================
# Builder
# =============================================================================

def build_transunet2d_v2(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
    use_deep_supervision: bool = True,
):
    return BoundaryAwareTransUNet2D_v2(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=32,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.0,
        use_boundary_head=use_boundary_head,
        use_deep_supervision=use_deep_supervision,
        window_size=4,
    )


if __name__ == "__main__":
    model = build_transunet2d_v2(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 128, 128)
    model.train()
    out = model(x)
    print("seg:", out["seg"].shape)
    if "ds2" in out:
        print("ds2:", out["ds2"].shape)
    if "boundary" in out:
        print("boundary:", out["boundary"].shape)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total / 1e6:.2f}M")
