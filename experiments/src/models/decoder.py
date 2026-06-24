import torch
import torch.nn as nn
from .onsampling import Onsampling3D
from .dsa_block import SABlock3D
from .attention import SkipAttentionBlock3D

class UnetrUpBlockWithAttention3D(nn.Module):
    """
    Полный блок декодера Swin DER:
    Onsampling → SCP AG → Concat → DSA Block
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: str = "onsampling",
        sa_block: bool = True,
    ):
        super().__init__()

        # Upsampling
        if upsample == 'onsampling':
            self.upsample = Onsampling3D(
                in_channels=in_channels,
                out_channels=out_channels,
                dyscope=True,
            )
        elif upsample == 'transconv':
            self.upsample = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            raise ValueError(f"Unknown upsample: {upsample}")

        # Feature extraction после конкатенации
        if sa_block:
            self.conv_block = SABlock3D(
                in_channels=out_channels + out_channels,
                out_channels=out_channels,
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv3d(out_channels * 2, out_channels, 3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.01, inplace=True),
            )

        # Skip-connection attention gate
        self.skipatten = SkipAttentionBlock3D(
            F_g=out_channels,
            F_l=out_channels,
            F_int=in_channels,
        )

    def forward(self, inp, skip):
        out = self.upsample(inp)
        skip = self.skipatten(out, skip)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out