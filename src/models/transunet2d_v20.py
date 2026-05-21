from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transunet2d_v14 import BoundaryAwareTransUNet2D_v14


class _FrequencyAuxiliaryHead(nn.Module):
    """Training-time auxiliary head using image frequency cues and posterior cues."""

    def __init__(self, hidden_ch: int = 16):
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
        self.head = nn.Sequential(
            nn.Conv2d(8, hidden_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 1),
        )

    def _edge(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    def forward(self, image: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        low = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        high = gray - low
        image_edge = self._edge(gray)
        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        local_mean = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        local_contrast = (gray - local_mean).abs()
        prob = torch.sigmoid(logits)
        uncertainty = prob * (1.0 - prob)
        features = torch.cat(
            [gray, low, high.abs(), image_edge, lap, local_contrast, prob, uncertainty],
            dim=1,
        )
        return self.head(features)


class BoundaryAwareTransUNet2D_v20(BoundaryAwareTransUNet2D_v14):
    """v20: stable v14 inference path with frequency auxiliary training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frequency_aux_head = _FrequencyAuxiliaryHead(hidden_ch=16)

    def forward(self, x: torch.Tensor):
        outputs = super().forward(x)
        if self.training:
            outputs["freq_aux"] = self.frequency_aux_head(x, outputs["seg"])
        return outputs


def build_transunet2d_v20(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
    use_deep_supervision: bool = True,
):
    return BoundaryAwareTransUNet2D_v20(
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
    model = build_transunet2d_v20()
    x = torch.randn(2, 1, 224, 224)
    model.train()
    out = model(x)
    print("[Train] seg:", out["seg"].shape)
    print("[Train] freq_aux:", out["freq_aux"].shape)
    print("[Train] aux:", sorted(out.keys()))
    model.eval()
    print("[Eval] keys:", sorted(model(x).keys()))
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
