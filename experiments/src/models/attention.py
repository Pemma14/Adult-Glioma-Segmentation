import torch
import torch.nn as nn

class SkipAttentionBlock3D(nn.Module):
    """
    Spatial-Channel Parallel Attention Gate (SCP AG) для 3D.
    F_g: каналы decoder feature (gating signal)
    F_l: каналы encoder feature (skip connection)
    F_int: промежуточные каналы
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        # --- Spatial Attention ---
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        # --- Channel Attention ---
        self.avg_x = nn.AdaptiveAvgPool3d(1)
        self.avg_g = nn.AdaptiveAvgPool3d(1)
        self.L_x = nn.Linear(F_l, F_l)
        self.L_g = nn.Linear(F_g, F_l)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: decoder feature (gating signal), shape (B, F_g, D, H, W)
        x: encoder feature (skip), shape (B, F_l, D, H, W)
        """
        # Spatial attention
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # (B, 1, D, H, W)

        # Channel attention
        bx, cx, dx, hx, wx = x.size()
        bg, cg, dg, hg, wg = g.size()

        avg_pool_x = self.avg_x(x).view(bx, cx)
        avg_pool_g = self.avg_g(g).view(bg, cg)

        channel_att_x = self.L_x(avg_pool_x)
        channel_att_g = self.L_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).view(bx, cx, 1, 1, 1)

        # Комбинирование
        psi = psi * scale
        return x * psi