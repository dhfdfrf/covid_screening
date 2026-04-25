from __future__ import annotations

from typing import Sequence

from monai.networks.nets import DynUNet


def build_dynunet2d(in_channels: int = 1, out_channels: int = 1):
    """
    nnU-Net-like backbone in MONAI (DynUNet).
    Input : (B, 1, H, W)
    Output: (B, 1, H, W) raw logits
    """

    # 5-level topology ~ (32, 64, 128, 256, 512)
    # Product of strides = 16 -> H,W should be divisible by 16
    model = DynUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3, 3, 3, 3),
        strides=(1, 2, 2, 2, 2),
        upsample_kernel_size=(2, 2, 2, 2),
        filters=(32, 64, 128, 256, 512),
        deep_supervision=False,
        res_block=True,
    )
    return model
