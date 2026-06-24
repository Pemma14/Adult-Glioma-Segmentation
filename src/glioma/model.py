"""Model loading utilities for clinical glioma segmentation inference."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.models import get_model
from src.utils.model import load_model_weights
from src.glioma.settings import load_model_config

logger = logging.getLogger(__name__)


def load_model_for_inference(
    checkpoint_path: str | Path,
    device: torch.device,
    config: dict | None = None,
) -> torch.nn.Module:
    """Load a single model from the registry for inference.

    Args:
        checkpoint_path: Path to the checkpoint ``.pth`` file.
        device: Torch device to load the model onto.
        config: Model config dict. If ``None``, loaded from the active registry
            version via ``GLIOMA__MODEL_VERSION``.

    Returns:
        The model in eval mode.
    """
    if config is None:
        config = load_model_config()

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_name = config["model"]["model_name"]
    model = get_model(model_name, config["model"]).to(device)
    load_model_weights(model, checkpoint_path, device, model_name=model_name)
    model.eval()
    logger.info("Loaded model from %s", checkpoint_path)
    return model


def load_ensemble_for_inference(
    device: torch.device,
    config: dict | None = None,
    folds: list[int] | None = None,
) -> list[torch.nn.Module]:
    """Load ensemble members from the registry for inference.

    Args:
        device: Torch device to load models onto.
        config: Model config dict. If ``None``, loaded from the active registry
            version via ``GLIOMA__MODEL_VERSION``.
        folds: Specific fold indices to load. If ``None``, loads all folds
            defined in the registry config.

    Returns:
        List of models in eval mode.
    """
    if config is None:
        config = load_model_config()

    registry = config["registry"]
    checkpoint_dir = Path(registry["checkpoint_dir"])
    pattern = registry["checkpoint_pattern"]
    if folds is None:
        folds = registry["ensemble_folds"]

    models: list[torch.nn.Module] = []
    for fold in folds:
        checkpoint_path = checkpoint_dir / pattern.format(i=fold)
        model = load_model_for_inference(checkpoint_path, device, config)
        models.append(model)

    logger.info("Loaded ensemble with %d models", len(models))
    return models
