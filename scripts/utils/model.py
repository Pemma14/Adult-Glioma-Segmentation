import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import logging
from clearml import Task

logger = logging.getLogger(__name__)


# Mapping from the Python class name used in this project to the model_name
# configured in YAML files.
_CLASS_TO_MODEL_NAME = {
    "BaselineUNet": "unet3d",
    "SwinUNETR": "swin_unetr",
    "SwinDER3D": "swin_der",
}


def _model_arch_name(model: nn.Module) -> str:
    """Return the model_name string that corresponds to the given module."""
    return _CLASS_TO_MODEL_NAME.get(model.__class__.__name__, model.__class__.__name__)


def _detect_architecture_from_state_dict(state_dict: dict) -> str | None:
    """
    Infer the family of a checkpoint from its state_dict keys.

    Returns ``"unet3d"`` for plain MONAI UNet checkpoints or ``"swin"`` for
    any Swin-based architecture (SwinUNETR / SwinDER).  ``None`` means the
    architecture could not be determined.
    """
    if not state_dict:
        return None
    first_key = next(iter(state_dict))
    if first_key.startswith("model.model."):
        return "unet3d"
    if first_key.startswith(("swinViT.", "encoder", "decoder", "out")):
        return "swin"
    return None


def _checkpoint_path(model_name, fold, is_best=True):
    prefix = "best" if is_best else "last"
    return f"{prefix}_model_{model_name}_fold{fold}.pth"


def peek_task_id(path):
    """Извлекает clearml_task_id из чекпойнта без загрузки весов модели."""
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


def resolve_checkpoint_path(checkpoint_path: str | Path | None, clearml_model_id: str | None,
                            root_dir: Path | None = None) -> Path:
    """Resolve a checkpoint path from a local file or a ClearML model ID."""
    if checkpoint_path is not None and clearml_model_id is not None:
        raise ValueError("Specify either --checkpoint or --clearml_model_id, not both.")
    if checkpoint_path is None and clearml_model_id is None:
        raise ValueError("Either --checkpoint or --clearml_model_id must be provided.")

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            if root_dir is not None:
                path = root_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    # Lazy import: ClearML is only required when downloading a remote model.
    from clearml import Model

    logger.info("Downloading model %s from ClearML", clearml_model_id)
    model_obj = Model(model_id=clearml_model_id)
    path = model_obj.get_local_copy()
    logger.info("Model downloaded to %s", path)
    return Path(path)


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    model_name: str | None = None,
) -> None:
    """Load model weights from a checkpoint file.

    Performs a lightweight architecture sanity check so that a mismatch
    between a Swin checkpoint and a UNet model (or vice versa) produces an
    actionable error instead of a long PyTorch state_dict diff.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        checkpoint_model_name = checkpoint.get("model_name")
        if checkpoint_model_name is not None and model_name is not None and checkpoint_model_name != model_name:
            raise RuntimeError(
                f"Checkpoint was trained with model '{checkpoint_model_name}', "
                f"but the current config uses model '{model_name}'. "
                f"Use the matching config, e.g. --config configs/{checkpoint_model_name}.yaml."
            )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            epoch = checkpoint.get("epoch")
            best_dice = checkpoint.get("best_dice")
            logger.info("Loaded training checkpoint metadata: epoch=%s, best_dice=%s", epoch, best_dice)
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    checkpoint_arch = _detect_architecture_from_state_dict(state_dict)
    model_arch = _model_arch_name(model)
    if checkpoint_arch is not None and model_arch != checkpoint_arch:
        # "swin" is the broad family for SwinUNETR / SwinDER; only raise a
        # specific mismatch when one side is the plain UNet3D.
        if checkpoint_arch == "unet3d" or model_arch == "unet3d":
            raise RuntimeError(
                f"Architecture mismatch: checkpoint looks like '{checkpoint_arch}', "
                f"but the model built from config is '{model_arch}'. "
                f"Did you pass the wrong --config? For a '{checkpoint_arch}' checkpoint use "
                f"--config configs/{checkpoint_arch}.yaml."
            )
        logger.warning(
            "Checkpoint architecture ('%s') may differ from the configured model ('%s'). "
            "If loading fails, check that --config matches the checkpoint.",
            checkpoint_arch,
            model_arch,
        )

    model.load_state_dict(state_dict)


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
        "model_name": config["model_name"],
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
    """Загрузка чекпоинта для возобновления обучения."""
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
