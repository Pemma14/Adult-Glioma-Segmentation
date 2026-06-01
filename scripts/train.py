import os
import sys

# Добавляем корень проекта в путь для импорта моделей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric

from clearml import Task, Logger
from models import get_model

# Настройка ClearML
task = Task.init(project_name='AdultGliomaSegmentation', task_name='Model_Comparison')
logger = Logger.current_logger()

CONFIG = {
    "model_name": "swin_unetr",  # unet3d, swin_unetr, swin_der
    "data_dir": "./data/BraTS2021",
    "img_size": (128, 128, 128),
    "in_channels": 4,
    "out_channels": 3,  # BraTS classes: WT, TC, ET (обычно)
    "feature_size": 48,
    "batch_size": 1,
    "max_epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "val_interval": 5,
    "transfer_learning": True,
    "pretrained_path": "./pretrained/model_swinvit.pt",
    "deep_supervision": True,
}

task.connect(CONFIG)

def load_pretrained_weights(model, model_name, path):
    if not os.path.exists(path):
        print(f"Pretrained weights not found at {path}. Skipping.")
        return model
    
    print(f"Loading pretrained weights for {model_name} from {path}")
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

# --- Transforms ---
# (Упрощенно для примера, в реальности BraTS требует специфической склейки классов)
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=CONFIG["img_size"],
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
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])

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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Модель
    model = get_model(CONFIG["model_name"], CONFIG).to(device)
    
    # 2. Transfer Learning
    if CONFIG["transfer_learning"] and CONFIG["model_name"] in ["swin_unetr", "swin_der"]:
        model = load_pretrained_weights(model, CONFIG["model_name"], CONFIG["pretrained_path"])
    
    # 3. Loss & Optimizer
    dice_ce_loss = DiceCELoss(to_onehot_y=False, sigmoid=True) # BraTS labels are often multi-label
    if CONFIG["model_name"] == "swin_der" and CONFIG["deep_supervision"]:
        loss_function = DeepSupervisionLoss(dice_ce_loss)
    else:
        loss_function = dice_ce_loss
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    
    # 4. Data (Mock for structure)
    # train_ds = Dataset(data=train_files, transform=train_transforms)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    # train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=1)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    best_dice = 0
    for epoch in range(CONFIG["max_epochs"]):
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
        print(f"Epoch {epoch} completed.")

        if (epoch + 1) % CONFIG["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                # val_dice logic...
                val_dice = 0.8 # mock
                logger.report_scalar("Dice", "val", iteration=epoch, value=val_dice)
                
                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save(model.state_dict(), "best_model.pth")
                    task.update_output_model("best_model.pth", name=f"best_{CONFIG['model_name']}")

if __name__ == "__main__":
    train()
