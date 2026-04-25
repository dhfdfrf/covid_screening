from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PromptEncoder(nn.Module):
    """
    Minimal text encoder: UTF-8 bytes -> embedding -> mean pooling.
    This is intentionally lightweight so that your pipeline can run without HuggingFace/BERT downloads.
    """
    def __init__(self, embed_dim: int = 256, max_len: int = 64):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(256, embed_dim)
        self.proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU())

    @staticmethod
    def _to_byte_ids(prompt: str, max_len: int) -> torch.Tensor:
        b = prompt.encode("utf-8")[:max_len]
        ids = list(b) + [0] * (max_len - len(b))
        return torch.tensor(ids, dtype=torch.long)

    def build_prompt_buffer(self, prompt: str) -> torch.Tensor:
        return self._to_byte_ids(prompt, self.max_len)

    def forward(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        # prompt_ids: (B, L)
        x = self.emb(prompt_ids)          # (B, L, D)
        x = x.mean(dim=1)                 # (B, D)
        return self.proj(x)               # (B, D)


class LAVTGateUNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32, prompt: str = "covid-19 infection region"):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.text = _PromptEncoder(embed_dim=256, max_len=64)
        prompt_ids = self.text.build_prompt_buffer(prompt)
        self.register_buffer("prompt_ids", prompt_ids, persistent=True)

        # gates: text -> channel bias per skip
        self.txt_to_c1 = nn.Linear(256, c1)
        self.txt_to_c2 = nn.Linear(256, c2)
        self.txt_to_c3 = nn.Linear(256, c3)

        self.enc1 = _ConvBlock(in_channels, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.enc4 = _ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec3 = _ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = _ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = _ConvBlock(c1 + c1, c1)

        self.out = nn.Conv2d(c1, out_channels, 1)

    def _gate(self, feat: torch.Tensor, txt_bias: torch.Tensor) -> torch.Tensor:
        # feat: (B,C,H,W), txt_bias: (B,C)
        gate = torch.sigmoid(txt_bias).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        return feat * gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        prompt = self.prompt_ids.unsqueeze(0).repeat(b, 1)  # (B, L)
        t = self.text(prompt)                               # (B, 256)

        s1 = self.enc1(x)
        x = self.pool(s1)

        s2 = self.enc2(x)
        x = self.pool(s2)

        s3 = self.enc3(x)
        x = self.pool(s3)

        x = self.enc4(x)

        # text-conditioned gates (simplified language-aware conditioning)
        s1g = self._gate(s1, self.txt_to_c1(t))
        s2g = self._gate(s2, self.txt_to_c2(t))
        s3g = self._gate(s3, self.txt_to_c3(t))

        x = self.up3(x)
        x = torch.cat([x, s3g], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, s2g], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, s1g], dim=1)
        x = self.dec1(x)

        return self.out(x)


def build_lavt2d(in_channels: int = 1, out_channels: int = 1, prompt: str = "covid-19 infection region"):
    return LAVTGateUNet2D(in_channels=in_channels, out_channels=out_channels, base=32, prompt=prompt)
    