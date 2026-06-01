import math
import torch
import torch.nn as nn
from typing import Union, Sequence, Optional
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Conv

from mmcv.ops import deform_conv3d

class DeformableConvV3D(nn.Module):
    """
    3D Deformable Convolution V2 implementation using mmcv.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        stride: Union[Sequence[int], int] = 1,
        padding: Optional[Union[Sequence[int], int]] = None,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        deformable_groups: int = 1,
    ):
        super().__init__()
        self.spatial_dims = 3
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Нормализация параметров
        self.kernel_size = self._tuple(kernel_size)
        self.stride = self._tuple(stride)
        self.dilation = self._tuple(dilation)
        self.padding = self._get_padding(self.kernel_size, self.stride) if padding is None else self._tuple(padding)
        self.groups = groups
        self.deformable_groups = deformable_groups

        # Веса ядра для DCN
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Offset + mask: 4 канала на каждую точку ядра (3 смещения + 1 маска)
        out_ch_offset_mask = self.deformable_groups * (3 + 1) * math.prod(self.kernel_size)
        self.conv_offset_mask = Conv[Conv.CONV, 3](
            in_channels=in_channels,
            out_channels=out_ch_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        
        self.reset_parameters()

    def _tuple(self, v):
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return tuple(v)
        elif isinstance(v, int):
            return (v, v, v)
        raise ValueError(f"Invalid 3D param: {v}")

    def _get_padding(self, kernel_size, stride):
        return tuple((k - s + 1) // 2 for k, s in zip(kernel_size, stride))

    def forward(self, x):
        # Вычисляем смещения и маску
        offset_mask = self.conv_offset_mask(x)
        outputs = torch.chunk(offset_mask, 4, dim=1)  # 3 канала offsets + 1 канал mask
        offset = torch.cat(outputs[:3], dim=1)
        mask = torch.sigmoid(outputs[3])

        return deform_conv3d(
            input=x,
            offset=offset,
            alpha=mask,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            n_weight_groups=self.groups,
            n_offset_groups=self.deformable_groups,
        )

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


class ConvAttentionBlock3D(nn.Module):
    """
    Attention-ветвь для DSA Block.
    Downsample → Conv → ReLU → Conv → Upsample
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_name: str = "instance",
    ):
        super().__init__()
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)

        self.conv1 = Convolution(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, strides=1, act=None, norm=None, conv_only=False,
        )
        self.conv2 = Convolution(
            spatial_dims=3, in_channels=out_channels, out_channels=out_channels,
            kernel_size=kernel_size, strides=1, act=None, norm=None, conv_only=False,
        )

        self.lrelu = nn.LeakyReLU(0.01, inplace=True)

        if norm_name == "instance":
            self.norm1 = nn.InstanceNorm3d(out_channels)
            self.norm2 = nn.InstanceNorm3d(out_channels)
        else:
            self.norm1 = nn.BatchNorm3d(out_channels)
            self.norm2 = nn.BatchNorm3d(out_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, inp):
        atten = self.avgpool(inp)
        atten = self.conv1(atten)
        atten = self.norm1(atten)
        atten = self.lrelu(atten)
        atten = self.conv2(atten)
        atten = self.norm2(atten)
        atten = self.upsample(atten)
        return atten


class SABlock3D(nn.Module):
    """
    Deformable Squeeze-and-Attention Block (DSA Block) для декодера.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_name: str = "instance",
    ):
        super().__init__()
        self.conv = Convolution(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, strides=1, act=None, norm=None, conv_only=False,
        )
        self.deformconv = DeformableConvV3D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
        )
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)

        if norm_name == "instance":
            self.norm1 = nn.InstanceNorm3d(out_channels)
            self.norm2 = nn.InstanceNorm3d(out_channels)
        else:
            self.norm1 = nn.BatchNorm3d(out_channels)
            self.norm2 = nn.BatchNorm3d(out_channels)

        self.attenblock = ConvAttentionBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            norm_name=norm_name,
        )

    def forward(self, inp):
        atten = self.attenblock(inp)
        out = self.conv(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.deformconv(out)
        out = self.norm2(out)
        out = (atten * out) + atten
        out = self.lrelu(out)
        return out