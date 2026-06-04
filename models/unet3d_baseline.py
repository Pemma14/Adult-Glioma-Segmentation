import torch.nn as nn
from monai.networks.nets import UNet

class BaselineUNet(nn.Module):
    """
    Baseline 3D UNet model using MONAI implementation.
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    ):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
        )

    def forward(self, x):
        return self.model(x)
