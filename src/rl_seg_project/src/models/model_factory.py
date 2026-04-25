from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class ModelBuildConfig:
    name: str
    in_channels: int = 1
    out_channels: int = 1
    image_size: Optional[Tuple[int, int]] = None
    prompt: Optional[str] = None


def available_models() -> Sequence[str]:
    return (
        "unet2d",
        "attention_unet2d",
        "dynunet2d",
        "transunet2d",
        "transunet2d_v2",
        "transunet2d_v3",
        "swin_unet2d",
        "uctransnet2d",
        "lavt2d",
    )


_ALIASES: Dict[str, str] = {
    "unet": "unet2d",
    "attention_unet": "attention_unet2d",
    "attunet": "attention_unet2d",
    "nnunet": "dynunet2d",
    "dynunet": "dynunet2d",
    "transunet": "transunet2d",
    "transunet_v2": "transunet2d_v2",
    "transunet2dv2": "transunet2d_v2",
    "transunet_v3": "transunet2d_v3",
    "transunet2dv3": "transunet2d_v3",
    "swin-unet": "swin_unet2d",
    "swinunet": "swin_unet2d",
    "uctransnet": "uctransnet2d",
    "lavt": "lavt2d",
}


def build_model(cfg: ModelBuildConfig) -> nn.Module:
    name = cfg.name.strip().lower()
    name = _ALIASES.get(name, name)

    if name == "unet2d":
        from src.models.unet2d import build_unet2d
        return build_unet2d(in_channels=cfg.in_channels, out_channels=cfg.out_channels)

    if name == "attention_unet2d":
        from src.models.attention_unet2d import build_attention_unet2d
        return build_attention_unet2d(in_channels=cfg.in_channels, out_channels=cfg.out_channels)

    if name == "dynunet2d":
        from src.models.dynunet2d import build_dynunet2d
        return build_dynunet2d(in_channels=cfg.in_channels, out_channels=cfg.out_channels)

    if name == "transunet2d":
        from src.models.transunet2d import build_transunet2d
        return build_transunet2d(in_channels=cfg.in_channels, out_channels=cfg.out_channels)

    if name == "transunet2d_v2":
        from src.models.transunet2d_v2 import build_transunet2d_v2
        return build_transunet2d_v2(in_channels=cfg.in_channels, out_channels=cfg.out_channels)

    if name == "transunet2d_v3":
        from src.models.transunet2d_v3 import build_transunet2d_v3
        return build_transunet2d_v3(in_channels=cfg.in_channels, out_channels=cfg.out_channels)

    if name == "swin_unet2d":
        from src.models.swin_unet2d import build_swin_unet2d
        return build_swin_unet2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            image_size=cfg.image_size,
        )

    if name == "uctransnet2d":
        from src.models.uctransnet2d import build_uctransnet2d
        return build_uctransnet2d(in_channels=cfg.in_channels, out_channels=cfg.out_channels)

    if name == "lavt2d":
        from src.models.lavt2d import build_lavt2d
        return build_lavt2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            prompt=cfg.prompt or "covid-19 infection region",
        )

    raise ValueError(
        f"Unknown model name: {cfg.name}. "
        f"Available: {', '.join(available_models())}"
    )