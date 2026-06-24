import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.factories import Conv
from monai.networks.blocks.convolutions import Convolution
from monai.networks.utils import pixelshuffle


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Onsampling3D(nn.Module):
    """
    3D Onsampling: learnable trilinear interpolation with offsets.
    Scale factor = 2 (upsample в 2 раза по каждой оси).
    Для 3D используется 8 соседей (2x2x2).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 64,
        scale: int = 2,
        kernel_size_encoder: int = 3,
        dyscope: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.spatial_dims = 3

        # Если каналы не совпадают — проекция
        if in_channels != out_channels:
            self.preconv = Conv[Conv.CONV, 3](
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
            in_channels = out_channels

        # Ветвь вычисления весов соседей
        self.comp = Convolution(
            spatial_dims=3,
            in_channels=out_channels,
            out_channels=mid_channels,
            kernel_size=1,
            strides=1,
            act="RELU",
            norm="INSTANCE",
        )
        self.enc = Convolution(
            spatial_dims=3,
            in_channels=mid_channels,
            out_channels=(scale * 2) ** 3,  # 8 для scale=2
            kernel_size=kernel_size_encoder,
            strides=1,
            act=None,
            norm="INSTANCE",
        )

        # Learnable offsets: 3 * scale^3 = 24 для scale=2
        self.offset = Conv[Conv.CONV, 3](
            in_channels=in_channels,
            out_channels=self.spatial_dims * scale ** self.spatial_dims,
            kernel_size=1,
        )
        normal_init(self.offset, std=0.001)

        # Optional: dyscope ограничивает амплитуду offset
        if dyscope:
            self.scope = Conv[Conv.CONV, 3](
                in_channels=in_channels,
                out_channels=self.spatial_dims * scale ** self.spatial_dims,
                kernel_size=1,
            )
            constant_init(self.scope, val=0.0)

        # Фиксированные начальные позиции соседей
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        """Начальные позиции для 8 соседей в 3D."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h, h])).transpose(1, 3).reshape(1, -1, 1, 1, 1)

    def pixelshuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Кастомный pixelshuffle для 3D координат."""
        dim, factor = 3, self.scale
        input_size = list(x.size())
        keeped_dim = input_size[:-(dim + 1)]
        channels = input_size[-(dim + 1)]
        scale_divisor = factor ** dim

        spatial_start_idx = len(keeped_dim) + 1
        org_channels = int(channels // scale_divisor)
        output_size = keeped_dim + [org_channels] + [d * factor for d in input_size[spatial_start_idx:]]

        indices = list(range(spatial_start_idx, spatial_start_idx + 2 * dim))
        indices = indices[dim:] + indices[:dim]
        permute_indices = list(range(spatial_start_idx))
        for idx in range(dim):
            permute_indices.extend(indices[idx::dim])

        x = x.reshape(keeped_dim + [org_channels] + [factor] * dim + input_size[spatial_start_idx:])
        x = x.permute(permute_indices).reshape(output_size)
        return x

    def get_grid(self, x):
        """Генерация смещённой сетки координат."""
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos

        B, _, D, H, W = offset.shape
        offset = offset.view(B, 3, -1, D, H, W)

        coords_d = torch.arange(D, device=x.device) + 0.5
        coords_h = torch.arange(H, device=x.device) + 0.5
        coords_w = torch.arange(W, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij'))
        coords = coords.transpose(1, 3).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)

        grid = self.pixelshuffle(coords + offset)
        grid = grid.permute(0, 2, 3, 4, 5, 1).contiguous().flatten(0, 1)
        return grid

    def get_neighbor_pixels(self, x, grid):
        """Выборка 8 соседей для каждого пикселя сетки."""
        B, C, D, H, W = x.shape
        _, d_, h_, w_, _ = grid.shape

        # 8 соседей: все комбинации (-1,-1,-1) ... (1,1,1)
        offsets = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ], device=grid.device)

        coords = (grid.unsqueeze(1) + offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4))
        coords = coords.view(-1, d_, h_, w_, 3)

        # Нормализация в [-1, 1] для grid_sample
        normalizer = torch.tensor([W, H, D], dtype=x.dtype, device=x.device).view(1, 1, 1, 1, 3)
        coords = 2 * coords / normalizer - 1

        # Повторяем вход 8 раз и сэмплим
        X = F.grid_sample(
            x.repeat(8, 1, 1, 1, 1),
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode="border",
        )
        X = X.view(B, 8, C, d_, h_, w_).permute(0, 2, 1, 3, 4, 5)
        return X

    def forward(self, X):
        if hasattr(self, 'preconv'):
            X = self.preconv(X)

        # Вычисляем веса 8 соседей
        W = self.comp(X)
        W = self.enc(W)
        W = pixelshuffle(W, spatial_dims=3, scale_factor=self.scale)
        W = F.softmax(W, dim=1)  # (B, 8, sD, sH, sW)

        # Генерация смещённой сетки и выборка соседей
        grid = self.get_grid(X)
        X = self.get_neighbor_pixels(X, grid)

        # Взвешенная сумма: einsum по оси соседей
        X = torch.einsum('bkdhw,bckdhw->bcdhw', [W, X])
        return X