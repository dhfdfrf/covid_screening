from __future__ import annotations

from src.models.transunet2d_v11 import (
    BoundaryAwareTransUNet2D_v11,
    tta_inference,
)


class BoundaryAwareTransUNet2D_v12(BoundaryAwareTransUNet2D_v11):
    """v12 keeps the v11 architecture and is trained with deep_boundary loss."""


def build_transunet2d_v12(
    in_channels: int = 1,
    out_channels: int = 1,
    use_boundary_head: bool = True,
    use_deep_supervision: bool = True,
):
    return BoundaryAwareTransUNet2D_v12(
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
