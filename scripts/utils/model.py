import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import logging
from clearml import Task

logger = logging.getLogger(__name__)


def _checkpoint_path(model_name, fold, is_best=True):
    prefix = "best" if is_best else "last"
    return f"{prefix}_model_{model_name}_fold{fold}.pth"


def peek_task_id(path):
    """Извлекает clearml_task_id из чекпойнта без загрузки весов модели.

    Возвращает None, если файл не существует или task_id не сохранён.
    """
    if not path or not Path(path).exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning(f"Failed to peek task_id from {path}: {e}")
        return None
    if not isinstance(checkpoint, dict):
        return None
    return checkpoint.get("clearml_task_id")

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

def save_checkpoint(model, config, fold, optimizer=None, scheduler=None, scaler=None,
                    epoch=None, best_dice=None, is_best=True, save_rng=True):
    prefix = "best" if is_best else "last"
    save_path = _checkpoint_path(config["model_name"], fold, is_best=is_best)
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

    # Сохраняем RNG state для воспроизводимого resume
    if save_rng:
        checkpoint["rng_state"] = {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    # Сохраняем clearml task_id для продолжения того же эксперимента после resume
    current_task = Task.current_task()
    if current_task is not None:
        checkpoint["clearml_task_id"] = current_task.id

    torch.save(checkpoint, save_path)
    if current_task is not None:
        current_task.update_output_model(save_path, name=f"{prefix}_{config['model_name']}_fold{fold}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, restore_rng=True):
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

    # Восстановление RNG state для воспроизводимости после resume
    if restore_rng and "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        try:
            torch.set_rng_state(rng["torch"])
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            logger.info("RNG state restored from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to restore RNG state: {e}")

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
