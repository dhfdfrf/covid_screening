import torch.nn as nn
from monai.networks.nets import AttentionUnet


def build_attention_unet2d(in_channels: int = 1, out_channels: int = 1):
    """
    输入:  (B, 1, H, W)
    输出:  (B, 1, H, W) raw logits
    适配你当前 QaTa 的二分类分割任务
    """
    model = AttentionUnet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        dropout=0.0,
    )
    return model