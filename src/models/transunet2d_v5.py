from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TransUNet2D v5 — 在v4基础上继续精调

v4成果: Dice 0.7740 (首次超baseline), IoU 0.6699
v4不足: Precision 0.7689 仍低于baseline的 0.7862

v5分析与改进:
  1. 回归全注意力Transformer: bottleneck仅8x8=64 tokens, 
     全注意力O(64^2)=O(4096)完全无压力, 窗口化反而割裂全局上下文
  2. Skip Connection Dropout: 训练时随机丢弃部分skip特征,
     迫使decoder更保守/自立, 减少对encoder噪声特征的依赖 → 提升precision
  3. 移除边界精炼模块: v4的Sobel精炼可能引入额外噪声, 简化回1x1输出
  4. 增强Tversky Loss: alpha=0.7更强力地惩罚假阳性
  5. 更保守的deep supervision权重
"""


# =============================================================================
# 基础模块: 完全沿用baseline
# =============================================================================

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SpatialGate(nn.Module):
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

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return skip * self.psi(self.gate_proj(gate) + self.skip_proj(skip))


# =============================================================================
# 2D sin-cos positional encoding (与baseline相同)
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


# =============================================================================
# 改进1: Skip Connection Dropout
#   训练时随机丢弃skip特征通道, 让decoder不过度依赖encoder噪声 → 减少FP
# =============================================================================

class _SkipDropout(nn.Module):
    """训练时对skip connection施加channel-wise dropout."""
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        # Channel-wise dropout (比spatial dropout更适合skip connection)
        mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1], 1, 1),
                                          1.0 - self.p, device=x.device, dtype=x.dtype))
        return x * mask / (1.0 - self.p)


# =============================================================================
# 主模型
# =============================================================================

class BoundaryAwareTransUNet2D_v5(nn.Module):
    """
    v5 = baseline骨架 + 全注意力Transformer + Skip Dropout + Tversky Loss

    与baseline的差异 (仅3处):
      1. Transformer前使用pre-norm (更稳定的训练)
      2. Skip连接加channel dropout (正则化, 提升precision)
      3. 可选deep supervision (保守权重)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        skip_dropout: float = 0.1,
        use_boundary_head: bool = True,
        use_deep_supervision: bool = True,
    ):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels*2, base_channels*4, base_channels*8
        self.use_boundary_head = use_boundary_head
        self.use_deep_supervision = use_deep_supervision

        # ---- Encoder (baseline) ----
        self.enc1 = _ConvBlock(in_channels, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.enc4 = _ConvBlock(c3, c4)
        self.down = nn.MaxPool2d(2)

        # ---- 全注意力Transformer (回归baseline, 仅改用pre-norm) ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=c4,
            nhead=num_heads,
            dim_feedforward=c4 * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # pre-norm: 训练更稳定
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)
        self.tr_norm = nn.LayerNorm(c4)

        # ---- Decoder (baseline) ----
        self.up4 = nn.ConvTranspose2d(c4, c4, 2, stride=2)
        self.dec4 = _ConvBlock(c4 + c4, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _ConvBlock(c1 + c1, c1)

        # ---- Attention Gates (baseline) ----
        self.gate4 = _SpatialGate(c4, c4, max(c4//2, 1))
        self.gate3 = _SpatialGate(c3, c3, max(c3//2, 1))
        self.gate2 = _SpatialGate(c2, c2, max(c2//2, 1))
        self.gate1 = _SpatialGate(c1, c1, max(c1//2, 1))

        # ---- Skip Dropout (唯一新增模块) ----
        self.skip_drop4 = _SkipDropout(skip_dropout)
        self.skip_drop3 = _SkipDropout(skip_dropout)
        self.skip_drop2 = _SkipDropout(skip_dropout)
        self.skip_drop1 = _SkipDropout(skip_dropout)

        # ---- Output ----
        self.out = nn.Conv2d(c1, out_channels, 1)

        # ---- Deep supervision ----
        if self.use_deep_supervision:
            self.ds4 = nn.Conv2d(c4, out_channels, 1)
            self.ds3 = nn.Conv2d(c3, out_channels, 1)
            self.ds2 = nn.Conv2d(c2, out_channels, 1)

        # ---- Boundary head ----
        if self.use_boundary_head:
            self.boundary_head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor):
        orig_h, orig_w = x.shape[2], x.shape[3]

        # ---------- Encoder ----------
        s1 = self.enc1(x);  x = self.down(s1)
        s2 = self.enc2(x);  x = self.down(s2)
        s3 = self.enc3(x);  x = self.down(s3)
        s4 = self.enc4(x);  x = self.down(s4)

        # ---------- Full Attention Transformer ----------
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + _posenc_2d_sincos(h, w, c, tokens.device, tokens.dtype)
        tokens = self.tr_norm(tokens)
        tokens = self.transformer(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        # ---------- Decoder + Skip Dropout ----------
        x = self.up4(x)
        x = torch.cat([x, self.gate4(x, self.skip_drop4(s4))], 1)
        d4 = self.dec4(x)

        x = self.up3(d4)
        x = torch.cat([x, self.gate3(x, self.skip_drop3(s3))], 1)
        d3 = self.dec3(x)

        x = self.up2(d3)
        x = torch.cat([x, self.gate2(x, self.skip_drop2(s2))], 1)
        d2 = self.dec2(x)

        x = self.up1(d2)
        x = torch.cat([x, self.gate1(x, self.skip_drop1(s1))], 1)
        d1 = self.dec1(x)

        seg_logits = self.out(d1)
        outputs = {"seg": seg_logits}

        if self.use_deep_supervision and self.training:
            outputs["ds4"] = F.interpolate(self.ds4(d4), (orig_h, orig_w),
                                           mode="bilinear", align_corners=False)
            outputs["ds3"] = F.interpolate(self.ds3(d3), (orig_h, orig_w),
                                           mode="bilinear", align_corners=False)
            outputs["ds2"] = F.interpolate(self.ds2(d2), (orig_h, orig_w),
                                           mode="bilinear", align_corners=False)

        if self.use_boundary_head:
            outputs["boundary"] = self.boundary_head(d1)

        return outputs


# =============================================================================
# 改进2: Tversky Loss (更强FP惩罚)
# =============================================================================

class TverskyLoss(nn.Module):
    """alpha > beta → 更强惩罚假阳性 → 提升precision."""
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        p = pred.view(pred.size(0), -1)
        t = target.view(target.size(0), -1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class TverskyBCELoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 tversky_w: float = 0.5, bce_w: float = 0.5):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.bce = nn.BCEWithLogitsLoss()
        self.tw = tversky_w
        self.bw = bce_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.tw * self.tversky(pred, target) + self.bw * self.bce(pred, target)


class DeepSupervisionLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module | None = None,
                 weights: tuple = (1.0, 0.2, 0.1, 0.05)):
        super().__init__()
        self.loss_fn = loss_fn or TverskyBCELoss()
        self.weights = weights

    def forward(self, outputs: dict, target: torch.Tensor) -> torch.Tensor:
        loss = self.weights[0] * self.loss_fn(outputs["seg"], target)
        for i, key in enumerate(["ds2", "ds3", "ds4"], 1):
            if key in outputs:
                loss += self.weights[i] * self.loss_fn(outputs[key], target)
        return loss


# =============================================================================
# Builder
# =============================================================================

def build_transunet2d_v5(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
    use_deep_supervision: bool = True,
):
    return BoundaryAwareTransUNet2D_v5(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=32,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.0,
        skip_dropout=0.1,
        use_boundary_head=use_boundary_head,
        use_deep_supervision=use_deep_supervision,
    )


if __name__ == "__main__":
    model = build_transunet2d_v5(in_channels=1, out_channels=1)
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

    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    loss_fn = DeepSupervisionLoss()
    loss = loss_fn(out, target)
    print(f"Loss: {loss.item():.4f}")
