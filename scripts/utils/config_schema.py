"""Lightweight config schema and validation helpers.

This module intentionally has no heavy ML dependencies so it can be imported
by scripts that lazily load torch/monai.
"""
from __future__ import annotations


# Common keys that must be present in any merged training configuration.
REQUIRED_TRAINING_COMMON_KEYS = [
    "data_dir",
    "in_channels",
    "out_channels",
    "n_splits",
    "val_interval",
    "sw_batch_size",
    "batch_size",
    "img_size",
    "lr",
    "weight_decay",
    "max_epochs",
    "num_samples",
    "patience",
    "model_name",
    "deep_supervision",
    "transfer_learning",
]

# Model-specific architecture keys.
REQUIRED_TRAINING_ARCH_KEYS = {
    "swin_unetr": [
        "feature_size", "depths", "num_heads", "drop_rate",
        "attn_drop_rate", "dropout_path_rate", "use_checkpoint", "spatial_dims",
    ],
    "swin_der": [
        "feature_size", "depths", "num_heads", "drop_rate",
        "attn_drop_rate", "dropout_path_rate", "use_checkpoint", "spatial_dims",
    ],
    "unet3d": ["channels", "strides", "num_res_units", "norm"],
}

# Keys required for inference only (model-specific architecture keys are added dynamically).
REQUIRED_INFERENCE_COMMON_KEYS = [
    "data_dir",
    "in_channels",
    "out_channels",
    "img_size",
    "model_name",
]


def validate_config(config: dict, required_keys: list[str], context: str = "config") -> None:
    """Raise a clear error if any required key is missing from the config."""
    if not isinstance(config, dict):
        raise TypeError(f"{context} должен быть dict, получен {type(config).__name__}")

    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(
            f"Критическая ошибка: в {context} отсутствуют обязательные параметры: {missing}. "
            f"Проверьте файлы configs/base.yaml и configs/<model>.yaml."
        )


def get_required_training_keys(model_name: str) -> list[str]:
    """Return the full list of required training keys for a given model."""
    arch_keys = REQUIRED_TRAINING_ARCH_KEYS.get(model_name, [])
    if not arch_keys:
        raise KeyError(
            f"Критическая ошибка: неизвестная модель '{model_name}'. "
            f"Доступные модели: {list(REQUIRED_TRAINING_ARCH_KEYS.keys())}. "
            f"Добавьте архитектурные ключи в REQUIRED_TRAINING_ARCH_KEYS."
        )
    return REQUIRED_TRAINING_COMMON_KEYS + arch_keys


def get_required_inference_keys(model_name: str) -> list[str]:
    """Return the full list of required inference keys for a given model."""
    arch_keys = REQUIRED_TRAINING_ARCH_KEYS.get(model_name, [])
    if not arch_keys:
        raise KeyError(
            f"Критическая ошибка: неизвестная модель '{model_name}'. "
            f"Доступные модели: {list(REQUIRED_TRAINING_ARCH_KEYS.keys())}."
        )
    return REQUIRED_INFERENCE_COMMON_KEYS + arch_keys
