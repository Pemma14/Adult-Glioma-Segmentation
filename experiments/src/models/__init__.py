import torch
import torch.nn as nn
from .swin_unetr import SwinUNETR
from .unet3d_baseline import BaselineUNet

try:
    from .swin_der import SwinDER3D
except ImportError as _swin_der_import_error:
    SwinDER3D = None  # tvdcn is optional and required only for swin_der

def get_model(model_name, config):
    if model_name == "unet3d":
        return BaselineUNet(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=config["channels"],
            strides=config["strides"],
            num_res_units=config["num_res_units"],
            norm=config["norm"],
        )
    elif model_name == "swin_unetr":
        return SwinUNETR(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            feature_size=config["feature_size"],
            deep_supervision=config["deep_supervision"],
            use_checkpoint=config["use_checkpoint"],
            norm_name=config["norm_name"],
            drop_rate=config["drop_rate"],
            attn_drop_rate=config["attn_drop_rate"],
            dropout_path_rate=config["dropout_path_rate"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            normalize=config["normalize"],
        )
    elif model_name == "swin_der":
        if SwinDER3D is None:
            raise ImportError(
                "swin_der requires tvdcn. Install with: uv sync --extra gpu"
            )
        return SwinDER3D(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            feature_size=config["feature_size"],
            deep_supervision=config["deep_supervision"],
            use_checkpoint=config["use_checkpoint"],
            norm_name=config["norm_name"],
            drop_rate=config["drop_rate"],
            attn_drop_rate=config["attn_drop_rate"],
            dropout_path_rate=config["dropout_path_rate"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            normalize=config["normalize"],
            upsample=config["upsample"],
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
