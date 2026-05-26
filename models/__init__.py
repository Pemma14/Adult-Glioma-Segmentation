import torch
import torch.nn as nn
from monai.networks.nets import UNet, SwinUNETR
from .swin_der_3D import SwinDER3D

def get_model(model_name, config):
    if model_name == "unet3d":
        return UNet(
            spatial_dims=3,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
        )
    elif model_name == "swin_unetr":
        return SwinUNETR(
            img_size=config["img_size"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            feature_size=config["feature_size"],
            use_checkpoint=config.get("use_checkpoint", False),
        )
    elif model_name == "swin_der":
        return SwinDER3D(
            img_size=config["img_size"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            feature_size=config["feature_size"],
            deep_supervision=config.get("deep_supervision", True),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
