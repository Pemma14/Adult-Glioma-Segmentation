from pathlib import Path
import sys
import argparse
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_log = logging.getLogger(__name__)

# Добавляем корень проекта в путь для импорта моделей
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    MapTransform,
    ToTensord,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentiled,
    SpatialPadd,
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric

from clearml import Task, Logger
from models import get_model

def load_config(config_path, base_config_path="configs/base.yaml"):
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if config_path:
        with open(config_path, "r") as f:
            specific_config = yaml.safe_load(f)
            if specific_config:
                config.update(specific_config)
    
    return config

def get_transforms(config):
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ScaleIntensityRangePercentiled(
            keys="image",
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            channel_wise=True,
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=config["img_size"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config["img_size"],
            pos=1,
            neg=1,
            num_samples=4,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ScaleIntensityRangePercentiled(
            keys="image",
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            channel_wise=True,
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "label"], spatial_size=config["img_size"]),
        ToTensord(keys=["image", "label"]),
    ])
    return train_transforms, val_transforms

def load_pretrained_weights(model, model_name, path):
    if not Path(path).exists():
        logger_log.warning(f"Pretrained weights not found at {path}. Skipping.")
        return model
    
    logger_log.info(f"Loading pretrained weights for {model_name} from {path}")
    checkpoint = torch.load(path, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if model_name == "swin_unetr":
        model.load_state_dict(state_dict, strict=False)
    elif model_name == "swin_der":
        # Загружаем только в энкодер
        model.swinViT.load_state_dict(state_dict, strict=False)
    
    return model

# --- Transforms moved to get_transforms() ---

class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, outputs, target):
        if isinstance(outputs, (list, tuple)):
            total_loss = 0
            weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
            for i, output in enumerate(outputs):
                # Interpolate target to match output size
                if output.shape[2:] != target.shape[2:]:
                    curr_target = nn.functional.interpolate(target, size=output.shape[2:], mode='nearest')
                else:
                    curr_target = target
                total_loss += weights[i] * self.base_loss(output, curr_target)
            return total_loss
        return self.base_loss(outputs, target)

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger = Logger.current_logger()
    
    # 1. Transforms
    train_transforms, val_transforms = get_transforms(config)
    
    # 2. Модель
    model = get_model(config["model_name"], config).to(device)
    
    # 3. Transfer Learning
    if config["transfer_learning"] and config["model_name"] in ["swin_unetr", "swin_der"]:
        model = load_pretrained_weights(model, config["model_name"], config["pretrained_path"])
    
    # 4. Loss & Optimizer
    dice_ce_loss = DiceCELoss(to_onehot_y=False, sigmoid=True) # BraTS labels are often multi-label
    if config["model_name"] == "swin_der" and config.get("deep_supervision"):
        loss_function = DeepSupervisionLoss(dice_ce_loss)
    else:
        loss_function = dice_ce_loss
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    
    # 5. Data (Mock for structure)
    # train_ds = Dataset(data=train_files, transform=train_transforms)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    # train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=1)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    best_dice = 0
    for epoch in range(config["max_epochs"]):
        model.train()
        epoch_loss = 0
        # for batch_data in train_loader:
        #     inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        #     optimizer.zero_grad()
        #     outputs = model(inputs)
        #     loss = loss_function(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
        #     epoch_loss += loss.item()
        
        # logger.report_scalar("Loss", "train", iteration=epoch, value=epoch_loss)
        logger_log.info(f"Epoch {epoch} completed.")

        if (epoch + 1) % config["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                # val_dice logic...
                val_dice = 0.8 # mock
                logger.report_scalar("Dice", "val", iteration=epoch, value=val_dice)
                
                if val_dice > best_dice:
                    best_dice = val_dice
                    save_path = f"best_model_{config['model_name']}.pth"
                    torch.save(model.state_dict(), save_path)
                    Task.current_task().update_output_model(save_path, name=f"best_{config['model_name']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Adult Glioma Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/swin_unetr.yaml", help="Path to the specific config file")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Path to the base config file")
    args = parser.parse_args()

    # Загружаем конфигурацию
    config = load_config(args.config, args.base_config)

    # Настройка ClearML
    task = Task.init(
        project_name='AdultGliomaSegmentation', 
        task_name=f'Train_{config["model_name"]}',
        task_type=Task.TaskTypes.training
    )
    task.connect(config)

    train(config)
