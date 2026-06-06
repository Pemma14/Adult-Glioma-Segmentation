import torch
import torch.nn as nn
from pathlib import Path
import logging
from clearml import Task

logger = logging.getLogger(__name__)

def load_pretrained_weights(model, model_name, path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Pretrained weights not found at {path}")
    
    logger.info(f"Loading pretrained weights for {model_name} from {path}")
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

def save_checkpoint(model, config, fold, optimizer=None, scheduler=None, scaler=None, epoch=None, best_dice=None, is_best=True):
    prefix = "best" if is_best else "last"
    save_path = f"{prefix}_model_{config['model_name']}_fold{fold}.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, save_path)
    Task.current_task().update_output_model(save_path, name=f"{prefix}_{config['model_name']}_fold{fold}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Загрузка чекпоинта для возобновления обучения.
    
    Поддерживает как новый формат (dict с model_state_dict),
    так и старый формат (только state_dict модели).
    
    Returns:
        tuple: (start_epoch, best_dice) — эпоха для продолжения и лучший dice.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location="cpu")
    
    # Поддержка старого формата (только state_dict модели)
    if "model_state_dict" not in checkpoint:
        logger.info("Old checkpoint format detected (model state_dict only)")
        model.load_state_dict(checkpoint)
        return 0, 0.0
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    start_epoch = checkpoint.get("epoch", -1) + 1
    best_dice = checkpoint.get("best_dice", 0.0)
    
    logger.info(f"Resumed from epoch {checkpoint.get('epoch', '?')}, best_dice={best_dice:.4f}")
    return start_epoch, best_dice

class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, outputs, target):
        if isinstance(outputs, (list, tuple)):
            total_loss = 0
            # Normalized weights (standard for nnU-Net)
            # sum of weights = 1.0 + 0.5 + 0.25 + 0.125 + 0.0625 = 1.9375
            # but usually we use relative weights
            raw_weights = [1.0 / (2**i) for i in range(len(outputs))]
            weights = [w / sum(raw_weights) for w in raw_weights]
            
            for i, output in enumerate(outputs):
                if isinstance(target, (list, tuple)):
                    curr_target = target[i] if i < len(target) else target[-1]
                else:
                    curr_target = target
                
                # Interpolate target if size doesn't match
                if curr_target.shape[2:] != output.shape[2:]:
                    curr_target = nn.functional.interpolate(curr_target, size=output.shape[2:], mode='nearest')
                
                total_loss += weights[i] * self.base_loss(output, curr_target)
            return total_loss
        
        if isinstance(target, (list, tuple)):
            target = target[0]
        return self.base_loss(outputs, target)
