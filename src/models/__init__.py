import torch
import torch.nn as nn
from .swin_unetr import SwinUNETR


def get_model(model_name, config):
    if model_name == "swin_unetr":
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
