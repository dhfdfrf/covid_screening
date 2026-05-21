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
        "swin_unet2d",
        "uctransnet2d",
        "lavt2d",
        "transunet2d_v13",
        "transunet2d_v14",
        "transunet2d_v16",
        "transunet2d_v17",
        "transunet2d_v18",
        "transunet2d_v19",
        "transunet2d_v20",
        "lowdice_refinenet2d",
        "lowdice_prior_refinenet2d",
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
    "swin-unet": "swin_unet2d",
    "swinunet": "swin_unet2d",
    "uctransnet": "uctransnet2d",
    "lavt": "lavt2d",
    "transunet_v13": "transunet2d_v13",
    "transunet2dv13": "transunet2d_v13",
    "transunet_v14": "transunet2d_v14",
    "transunet2dv14": "transunet2d_v14",
    "transunet_v16": "transunet2d_v16",
    "transunet2dv16": "transunet2d_v16",
    "transunet_v17": "transunet2d_v17",
    "transunet2dv17": "transunet2d_v17",
    "transunet_v18": "transunet2d_v18",
    "transunet2dv18": "transunet2d_v18",
    "transunet_v19": "transunet2d_v19",
    "transunet2dv19": "transunet2d_v19",
    "transunet_v20": "transunet2d_v20",
    "transunet2dv20": "transunet2d_v20",
    "lowdice_refinenet": "lowdice_refinenet2d",
    "lowdice": "lowdice_refinenet2d",
    "lowdice_specialist": "lowdice_refinenet2d",
    "lowdice_prior_refinenet": "lowdice_prior_refinenet2d",
    "lowdice_prior": "lowdice_prior_refinenet2d",
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

    if name == "transunet2d_v13":
        from src.models.transunet2d_v13 import build_transunet2d_v13
        return build_transunet2d_v13(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
        )

    if name == "transunet2d_v14":
        from src.models.transunet2d_v14 import build_transunet2d_v14
        return build_transunet2d_v14(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
        )

    if name == "transunet2d_v16":
        from src.models.transunet2d_v16 import build_transunet2d_v16
        return build_transunet2d_v16(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
        )

    if name == "transunet2d_v17":
        from src.models.transunet2d_v17 import build_transunet2d_v17
        return build_transunet2d_v17(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
        )

    if name == "transunet2d_v18":
        from src.models.transunet2d_v18 import build_transunet2d_v18
        return build_transunet2d_v18(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
        )

    if name == "transunet2d_v19":
        from src.models.transunet2d_v19 import build_transunet2d_v19
        return build_transunet2d_v19(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
        )

    if name == "transunet2d_v20":
        from src.models.transunet2d_v20 import build_transunet2d_v20
        return build_transunet2d_v20(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
        )

    if name == "lowdice_refinenet2d":
        from src.models.lowdice_refinenet2d import build_lowdice_refinenet2d
        return build_lowdice_refinenet2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
        )

    if name == "lowdice_prior_refinenet2d":
        from src.models.lowdice_refinenet2d import build_lowdice_prior_refinenet2d
        return build_lowdice_prior_refinenet2d(
            out_channels=cfg.out_channels,
        )

    raise ValueError(
        f"Unknown model name: {cfg.name}. "
        f"Available: {', '.join(available_models())}"
    )
