from __future__ import annotations

from typing import Optional, Tuple

from monai.networks.nets import SwinUNETR


def build_swin_unet2d(
    in_channels: int = 1,
    out_channels: int = 1,
    image_size: Optional[Tuple[int, int]] = None,
):
    """
    Practical Swin-UNet-like model via MONAI SwinUNETR (2D).
    Input : (B, 1, H, W)
    Output: (B, 1, H, W) raw logits

    Note:
      - Many implementations require H,W divisible by 32 (2**5).
      - MONAI SwinUNETR requires feature_size divisible by 12.
    """

    img_size = image_size or (512, 512)

    try:
        return SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=24,   # 必须能被 12 整除
            use_checkpoint=True,
            spatial_dims=2,
        )
    except TypeError:
        return SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=24,   # 必须能被 12 整除
            use_checkpoint=True,
            spatial_dims=2,
        )