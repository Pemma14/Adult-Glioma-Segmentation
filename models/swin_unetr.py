import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from monai.networks.nets.swin_unetr import SwinTransformer

class SwinUNETR(nn.Module):
    """
    Custom implementation of Swin UNETR with Deep Supervision support.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 48,
        depths: tuple = (2, 2, 2, 2),
        num_heads: tuple = (3, 6, 12, 24),
        normalize: bool = True,
        use_checkpoint: bool = False,
        deep_supervision: bool = False,
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
    ):
        super().__init__()
        self.normalize = normalize
        self.deep_supervision = deep_supervision
        
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        
        if self.deep_supervision:
            self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=2 * feature_size, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=4 * feature_size, out_channels=out_channels)
            self.out4 = UnetOutBlock(spatial_dims=3, in_channels=8 * feature_size, out_channels=out_channels)

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        
        output = self.out(out)
        if self.deep_supervision and self.training:
            output1 = self.out1(dec0)
            output2 = self.out2(dec1)
            output3 = self.out3(dec2)
            output4 = self.out4(dec3)
            return [output, output1, output2, output3, output4]
        
        return output
