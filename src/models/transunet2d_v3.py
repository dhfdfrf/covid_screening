from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 改进1: 残差双卷积块 + SE注意力 (替代MSConv)
#   - MSConv的多分支并行可能引入噪声,降低精度
#   - 改用残差连接 + SE通道注意力,更稳定高效
# =============================================================================

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class _ResConvBlock(nn.Module):
    """残差双卷积块 + SE注意力: 稳定训练,提升精度."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
        )
        self.se = _SEBlock(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.se(out)
        return self.act(out + self.skip(x))


# =============================================================================
# 改进2: 改进的注意力门控 (替代CSAG)
#   - CSAG过于复杂(同时做通道+空间注意力),参数多但效果差
#   - 改用经典Attention Gate + 可学习残差缩放
# =============================================================================

class _AttentionGate(nn.Module):
    """Attention Gate with learnable residual scaling."""
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_x = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.norm = nn.InstanceNorm2d(inter_ch)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # 可学习的残差缩放: 初始化接近1,让训练初期skip几乎直通
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_g(gate)
        x = self.W_x(skip)
        attn = self.psi(F.leaky_relu(self.norm(g + x), 0.01))
        # 残差连接: skip * (1 - alpha + alpha * attn), 初期接近skip本身
        return skip * (1.0 - self.alpha + self.alpha * attn)


# =============================================================================
# 改进3: 相对位置编码的窗口Transformer (替代固定sincos)
#   - 可学习的相对位置偏置,对小图更有效
#   - 加入DropPath正则化
# =============================================================================

class _DropPath(nn.Module):
    """Stochastic depth for regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class _WindowedTransformerBlock(nn.Module):
    """Transformer block with windowed attention + relative position bias + DropPath."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, drop: float,
                 drop_path: float = 0.0, shift: bool = False, win_size: int = 4):
        super().__init__()
        self.shift = shift
        self.win_size = win_size
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(drop),
        )
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # 可学习相对位置偏置
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * win_size - 1) * (2 * win_size - 1), nhead)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        self._register_rel_pos_index(win_size)

    def _register_rel_pos_index(self, ws: int):
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij")).flatten(1)  # 2, ws*ws
        rel = coords[:, :, None] - coords[:, None, :]  # 2, N, N
        rel[0] += ws - 1
        rel[1] += ws - 1
        rel[0] *= 2 * ws - 1
        index = rel.sum(0)  # N, N
        self.register_buffer("rel_pos_index", index.long(), persistent=False)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        return self.rel_pos_bias[self.rel_pos_index.view(-1)].view(
            self.win_size * self.win_size, self.win_size * self.win_size, -1
        ).permute(2, 0, 1).unsqueeze(0)  # 1, nhead, N, N

    def _window_partition(self, x: torch.Tensor, h: int, w: int):
        ws = self.win_size
        b, _, c = x.shape
        x = x.view(b, h, w, c)
        if self.shift:
            x = torch.roll(x, shifts=(-ws // 2, -ws // 2), dims=(1, 2))
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
        return x[:, :h, :w, :].reshape(b, h * w, c)

    def _windowed_attn(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b_orig = x.shape[0]
        xw, hp, wp = self._window_partition(x, h, w)
        B, N, C = xw.shape
        qkv = self.qkv(xw).reshape(B, N, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn + self._get_rel_pos_bias()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self._window_unpartition(out, hp, wp, h, w, b_orig)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.drop_path(self._windowed_attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class _WindowedTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float,
                 drop_path: float = 0.1, win_size: int = 4):
        super().__init__()
        ffn_dim = d_model * 4
        dpr = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            _WindowedTransformerBlock(
                d_model, nhead, ffn_dim, dropout,
                drop_path=dpr[i],
                shift=(i % 2 == 1),
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
# 改进4: 输出精炼模块
#   - 在最终输出前加一个小型精炼模块,提升边界精度
# =============================================================================

class _RefinementModule(nn.Module):
    """轻量精炼模块: 用低层特征精炼分割边界."""
    def __init__(self, seg_ch: int, low_ch: int, out_ch: int):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, seg_ch, 1, bias=False),
            nn.InstanceNorm2d(seg_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(seg_ch * 2, seg_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(seg_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(seg_ch, out_ch, 1),
        )

    def forward(self, seg_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        low = self.low_proj(low_feat)
        return self.refine(torch.cat([seg_feat, low], dim=1))


# =============================================================================
# Main Model: BoundaryAwareTransUNet2D v3
# =============================================================================

class BoundaryAwareTransUNet2D_v3(nn.Module):
    """
    v3 改进要点 (针对v2精度下降问题):
    1. ResConvBlock+SE 替代 MSConv: 减少多分支噪声,残差连接稳定训练
    2. 简化AttentionGate + 可学习残差缩放: 训练初期skip直通,避免信息丢失
    3. 相对位置偏置 + DropPath: 更好的位置编码,正则化防过拟合
    4. 输出精炼模块: 用encoder低层特征精炼边界
    5. LeakyReLU: 避免神经元死亡
    """

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
    ):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        bottleneck_c = c4

        self.use_boundary_head = use_boundary_head
        self.use_deep_supervision = use_deep_supervision

        # ---- Encoder (ResConv + SE) ----
        self.enc1 = _ResConvBlock(in_channels, c1)
        self.enc2 = _ResConvBlock(c1, c2)
        self.enc3 = _ResConvBlock(c2, c3)
        self.enc4 = _ResConvBlock(c3, c4)
        self.down = nn.MaxPool2d(2)

        # ---- Windowed Transformer bottleneck ----
        self.transformer = _WindowedTransformerEncoder(
            d_model=bottleneck_c,
            nhead=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            drop_path=drop_path,
            win_size=window_size,
        )
        # 可学习位置嵌入 (替代固定sincos)
        self.pos_embed = None  # 延迟初始化

        # ---- Decoder ----
        self.up4 = nn.ConvTranspose2d(bottleneck_c, c4, 2, stride=2)
        self.dec4 = _ResConvBlock(c4 + c4, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _ResConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _ResConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _ResConvBlock(c1 + c1, c1)

        # ---- Attention Gates (简化版) ----
        self.gate4 = _AttentionGate(c4, c4, max(c4 // 2, 1))
        self.gate3 = _AttentionGate(c3, c3, max(c3 // 2, 1))
        self.gate2 = _AttentionGate(c2, c2, max(c2 // 2, 1))
        self.gate1 = _AttentionGate(c1, c1, max(c1 // 2, 1))

        # ---- 输出精炼模块 ----
        self.refinement = _RefinementModule(c1, c1, out_channels)

        # ---- Main output (用于deep supervision对齐) ----
        self.out = nn.Conv2d(c1, out_channels, 1)

        # ---- Deep supervision heads ----
        if self.use_deep_supervision:
            self.ds4 = nn.Conv2d(c4, out_channels, 1)
            self.ds3 = nn.Conv2d(c3, out_channels, 1)
            self.ds2 = nn.Conv2d(c2, out_channels, 1)

        # ---- Boundary head ----
        if self.use_boundary_head:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(c1, c1 // 2, 3, padding=1, bias=False),
                nn.InstanceNorm2d(c1 // 2),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(c1 // 2, 1, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        orig_h, orig_w = x.shape[2], x.shape[3]

        # ---------- Encoder ----------
        s1 = self.enc1(x);   x = self.down(s1)
        s2 = self.enc2(x);   x = self.down(s2)
        s3 = self.enc3(x);   x = self.down(s3)
        s4 = self.enc4(x);   x = self.down(s4)

        # ---------- Windowed Transformer ----------
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # B, H*W, C
        # 不加全局位置编码, 依赖transformer内部的相对位置偏置
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

        # ---------- 精炼输出 ----------
        seg_logits = self.refinement(d1, s1)  # 用enc1特征精炼

        outputs = {"seg": seg_logits}

        # Deep supervision
        if self.use_deep_supervision and self.training:
            outputs["ds4"] = F.interpolate(self.ds4(d4), size=(orig_h, orig_w),
                                           mode="bilinear", align_corners=False)
            outputs["ds3"] = F.interpolate(self.ds3(d3), size=(orig_h, orig_w),
                                           mode="bilinear", align_corners=False)
            outputs["ds2"] = F.interpolate(self.ds2(d2), size=(orig_h, orig_w),
                                           mode="bilinear", align_corners=False)

        if self.use_boundary_head:
            outputs["boundary"] = self.boundary_head(d1)

        return outputs


# =============================================================================
# 改进5: 组合损失函数 (Dice + Focal)
#   - 纯BCE容易被简单样本主导,Focal Loss关注难分样本
#   - Dice Loss直接优化目标指标
# =============================================================================

class DiceFocalLoss(nn.Module):
    """Dice + Focal组合损失, 提升精度和Dice指标."""

    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.dice_w = dice_weight
        self.focal_w = focal_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_w * self._dice_loss(pred, target) + self.focal_w * self._focal_loss(pred, target)


class DeepSupervisionLoss(nn.Module):
    """Deep supervision with Dice+Focal loss."""

    def __init__(self, loss_fn: nn.Module | None = None, weights: tuple = (1.0, 0.4, 0.2, 0.1)):
        super().__init__()
        self.loss_fn = loss_fn or DiceFocalLoss()
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

def build_transunet2d_v3(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
    use_deep_supervision: bool = True,
):
    return BoundaryAwareTransUNet2D_v3(
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
    )


if __name__ == "__main__":
    model = build_transunet2d_v3(in_channels=1, out_channels=1)
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

    # 测试loss
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    loss_fn = DeepSupervisionLoss()
    loss = loss_fn(out, target)
    print(f"Loss: {loss.item():.4f}")
