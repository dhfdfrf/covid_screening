from monai.networks.nets import UNet

def build_unet2d(in_channels: int = 1, out_channels: int = 1):
    # 小一点的通道数，显存友好
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )