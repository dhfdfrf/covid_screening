from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TransUNet2D v8

v7教训: 
  1. Dice+BCE损失远不如Focal Tversky+BCE (val_dice 0.7333 vs v6的0.7957)
  2. TTA写在forward()里, eval()时自动触发, 干扰validation和模型选择

v8 = v6 (历史最佳训练表现) + TTA作为独立方法
  - 架构: 完全同v6 (baseline ConvBlock/SpatialGate + 窗口Transformer + 边界精炼)
  - 损失: 恢复Focal Tversky + BCE (v6验证最优)
  - 正则化: DropPath(0.1) + decoder dropout(0.05) + 负bias初始化
  - TTA: 独立方法 predict_with_tta(), 不影响 forward(), 仅在最终推理时手动调用

使用方式:
  训练: 正常 model(x)
  验证: 正常 model(x)  
  最终推理: model.predict_with_tta(x) 或 model(x) 都可以
"""


# =============================================================================
# 基础模块
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
# 窗口Transformer
# =============================================================================

class _DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.bernoulli(torch.full((x.shape[0], 1, 1), keep,
                                          device=x.device, dtype=x.dtype))
        return x * mask / keep


class _WindowedTransformerBlock(nn.Module):
    def __init__(self, d: int, nhead: int, ffn: int, drop: float,
                 drop_path: float, shift: bool, ws: int):
        super().__init__()
        self.shift = shift
        self.ws = ws
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop),
            nn.Linear(ffn, d), nn.Dropout(drop),
        )
        self.drop_path = _DropPath(drop_path)

    def _partition(self, x, h, w):
        ws = self.ws; b, _, c = x.shape
        x = x.view(b, h, w, c)
        if self.shift:
            x = torch.roll(x, (-ws // 2, -ws // 2), (1, 2))
        ph = (ws - h % ws) % ws
        pw = (ws - w % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph))
        hp, wp = h + ph, w + pw
        x = x.view(b, hp // ws, ws, wp // ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
        return x, hp, wp

    def _unpartition(self, x, hp, wp, h, w, b):
        ws = self.ws; c = x.shape[-1]
        x = x.view(b, hp // ws, wp // ws, ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
        if self.shift:
            x = torch.roll(x, (ws // 2, ws // 2), (1, 2))
        return x[:, :h, :w, :].reshape(b, h * w, c)

    def forward(self, x, h, w):
        b = x.shape[0]
        r = x
        xn = self.norm1(x)
        xw, hp, wp = self._partition(xn, h, w)
        xw, _ = self.attn(xw, xw, xw)
        xn = self._unpartition(xw, hp, wp, h, w, b)
        x = r + self.drop_path(xn)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class _WindowedTransformerEncoder(nn.Module):
    def __init__(self, d: int, nhead: int, nlayers: int, drop: float,
                 drop_path: float = 0.1, ws: int = 4):
        super().__init__()
        dpr = [drop_path * i / max(nlayers - 1, 1) for i in range(nlayers)]
        self.layers = nn.ModuleList([
            _WindowedTransformerBlock(d, nhead, d * 4, drop, dpr[i],
                                     shift=(i % 2 == 1), ws=ws)
            for i in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, h, w):
        for layer in self.layers:
            x = layer(x, h, w)
        return self.norm(x)


# =============================================================================
# 边界精炼
# =============================================================================

class _BoundaryRefinement(nn.Module):
    def __init__(self, feat_ch: int, out_ch: int):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(feat_ch + 2, feat_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(feat_ch, out_ch, 1)

    def forward(self, feat: torch.Tensor, coarse_seg: torch.Tensor) -> torch.Tensor:
        seg_prob = torch.sigmoid(coarse_seg)
        ex = F.conv2d(seg_prob, self.sobel_x, padding=1)
        ey = F.conv2d(seg_prob, self.sobel_y, padding=1)
        refined = self.edge_conv(torch.cat([feat, ex, ey], dim=1))
        return self.out_conv(refined)


# =============================================================================
# Spatial Dropout
# =============================================================================

class _SpatialDropout2d(nn.Module):
    def __init__(self, p: float = 0.05):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1], 1, 1),
                                          1.0 - self.p, device=x.device, dtype=x.dtype))
        return x * mask / (1.0 - self.p)


# =============================================================================
# 主模型
# =============================================================================

class BoundaryAwareTransUNet2D_v8(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        decoder_dropout: float = 0.05,
        use_boundary_head: bool = True,
        use_deep_supervision: bool = True,
        window_size: int = 4,
    ):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.use_boundary_head = use_boundary_head
        self.use_deep_supervision = use_deep_supervision

        # ---- Encoder ----
        self.enc1 = _ConvBlock(in_channels, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.enc4 = _ConvBlock(c3, c4)
        self.down = nn.MaxPool2d(2)

        # ---- Windowed Transformer ----
        self.transformer = _WindowedTransformerEncoder(
            d=c4, nhead=num_heads, nlayers=num_transformer_layers,
            drop=dropout, drop_path=drop_path, ws=window_size,
        )

        # ---- Decoder ----
        self.up4 = nn.ConvTranspose2d(c4, c4, 2, stride=2)
        self.dec4 = _ConvBlock(c4 + c4, c4)
        self.ddrop4 = _SpatialDropout2d(decoder_dropout)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _ConvBlock(c3 + c3, c3)
        self.ddrop3 = _SpatialDropout2d(decoder_dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _ConvBlock(c2 + c2, c2)
        self.ddrop2 = _SpatialDropout2d(decoder_dropout)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _ConvBlock(c1 + c1, c1)

        # ---- Attention Gates ----
        self.gate4 = _SpatialGate(c4, c4, max(c4 // 2, 1))
        self.gate3 = _SpatialGate(c3, c3, max(c3 // 2, 1))
        self.gate2 = _SpatialGate(c2, c2, max(c2 // 2, 1))
        self.gate1 = _SpatialGate(c1, c1, max(c1 // 2, 1))

        # ---- Output ----
        self.coarse_out = nn.Conv2d(c1, out_channels, 1)
        self.refine = _BoundaryRefinement(c1, out_channels)

        # ---- Deep supervision ----
        if self.use_deep_supervision:
            self.ds4 = nn.Conv2d(c4, out_channels, 1)
            self.ds3 = nn.Conv2d(c3, out_channels, 1)
            self.ds2 = nn.Conv2d(c2, out_channels, 1)

        # ---- Boundary head ----
        if self.use_boundary_head:
            self.boundary_head = nn.Conv2d(c1, 1, 1)

        self._init_output_bias()

    def _init_output_bias(self):
        for m in [self.coarse_out]:
            if m.bias is not None:
                nn.init.constant_(m.bias, -1.0)
        if self.use_deep_supervision:
            for m in [self.ds4, self.ds3, self.ds2]:
                if m.bias is not None:
                    nn.init.constant_(m.bias, -1.0)
        if self.use_boundary_head and self.boundary_head.bias is not None:
            nn.init.constant_(self.boundary_head.bias, -1.0)

    def forward(self, x: torch.Tensor):
        """标准前向传播, 训练和验证都用这个, TTA不在这里."""
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Encoder
        s1 = self.enc1(x);  x = self.down(s1)
        s2 = self.enc2(x);  x = self.down(s2)
        s3 = self.enc3(x);  x = self.down(s3)
        s4 = self.enc4(x);  x = self.down(s4)

        # Transformer
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens, h, w)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        # Decoder
        x = self.up4(x)
        x = torch.cat([x, self.gate4(x, s4)], 1)
        d4 = self.ddrop4(self.dec4(x))

        x = self.up3(d4)
        x = torch.cat([x, self.gate3(x, s3)], 1)
        d3 = self.ddrop3(self.dec3(x))

        x = self.up2(d3)
        x = torch.cat([x, self.gate2(x, s2)], 1)
        d2 = self.ddrop2(self.dec2(x))

        x = self.up1(d2)
        x = torch.cat([x, self.gate1(x, s1)], 1)
        d1 = self.dec1(x)

        # Output
        coarse = self.coarse_out(d1)
        seg_logits = coarse + self.refine(d1, coarse)

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

    @torch.no_grad()
    def predict_with_tta(self, x: torch.Tensor) -> torch.Tensor:
        """Test-Time Augmentation: 仅在最终推理时手动调用.
        
        用法:
            model.eval()
            logits = model.predict_with_tta(x)  # 返回平均logits
            pred = (torch.sigmoid(logits) > 0.5).float()
        """
        was_training = self.training
        self.eval()

        seg = self.forward(x)["seg"]
        seg_h = torch.flip(self.forward(torch.flip(x, [3]))["seg"], [3])
        seg_v = torch.flip(self.forward(torch.flip(x, [2]))["seg"], [2])
        seg_hv = torch.flip(self.forward(torch.flip(x, [2, 3]))["seg"], [2, 3])

        if was_training:
            self.train()

        return (seg + seg_h + seg_v + seg_hv) / 4.0


# =============================================================================
# Loss: Focal Tversky + BCE (v6验证最优, 原样恢复)
# =============================================================================

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.6, beta: float = 0.4,
                 gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        p = pred.view(pred.size(0), -1)
        t = target.view(target.size(0), -1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky).pow(self.gamma).mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, gamma: float = 0.75,
                 ft_w: float = 0.6, bce_w: float = 0.4):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce = nn.BCEWithLogitsLoss()
        self.ft_w = ft_w
        self.bce_w = bce_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ft_w * self.ft(pred, target) + self.bce_w * self.bce(pred, target)


class DeepSupervisionLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module | None = None,
                 weights: tuple = (1.0, 0.2, 0.1, 0.05)):
        super().__init__()
        self.loss_fn = loss_fn or CombinedLoss()
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

def build_transunet2d_v8(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
    use_deep_supervision: bool = True,
):
    return BoundaryAwareTransUNet2D_v8(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=32,
        num_transformer_layers=4,
        num_heads=8,
        dropout=0.0,
        drop_path=0.1,
        decoder_dropout=0.05,
        use_boundary_head=use_boundary_head,
        use_deep_supervision=use_deep_supervision,
        window_size=4,
    )


if __name__ == "__main__":
    model = build_transunet2d_v8(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 128, 128)

    # 训练模式
    model.train()
    out = model(x)
    print("[Train] seg:", out["seg"].shape)
    if "ds2" in out:
        print("[Train] ds2:", out["ds2"].shape)

    # 普通推理
    model.eval()
    out_eval = model(x)
    print("[Eval] seg:", out_eval["seg"].shape)

    # TTA推理 (独立方法)
    tta_logits = model.predict_with_tta(x)
    print("[TTA] seg:", tta_logits.shape)

    diff = (out_eval["seg"] - tta_logits).abs().mean().item()
    print(f"Eval vs TTA mean diff: {diff:.4f}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total / 1e6:.2f}M")

    model.train()
    out = model(x)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    loss_fn = DeepSupervisionLoss()
    loss = loss_fn(out, target)
    print(f"Loss: {loss.item():.4f}")
