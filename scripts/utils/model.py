import torch
import torch.nn as nn
from pathlib import Path
import logging
from clearml import Task

logger = logging.getLogger(__name__)

def load_pretrained_weights(model, model_name, path):
    if not Path(path).exists():
        logger.warning(f"Pretrained weights not found at {path}. Skipping.")
        return model
    
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

def save_checkpoint(model, config, fold):
    save_path = f"best_model_{config['model_name']}_fold{fold}.pth"
    torch.save(model.state_dict(), save_path)
    Task.current_task().update_output_model(save_path, name=f"best_{config['model_name']}_fold{fold}")

class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, outputs, target):
        if isinstance(outputs, (list, tuple)):
            total_loss = 0
            weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
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
