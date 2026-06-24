from __future__ import annotations

import os
from pathlib import Path

import yaml

REGISTRY_DIR = Path("models_registry")


def get_model_version() -> str:
    version = os.getenv("GLIOMA__MODEL_VERSION")
    if not version:
        raise EnvironmentError(
            "GLIOMA__MODEL_VERSION environment variable is not set. "
            "Available versions can be found in the models_registry directory."
        )
    return version


def get_registry_dir() -> Path:
    return REGISTRY_DIR


def get_model_config_path(version: str | None = None) -> Path:
    if version is None:
        version = get_model_version()
    return get_registry_dir() / version / "config.yaml"


def load_model_config(version: str | None = None) -> dict:
    config_path = get_model_config_path(version)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found for version '{version}': {config_path}. "
            f"Make sure the version exists in {get_registry_dir()}."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {config_path}: expected dict, got {type(config).__name__}")

    return config
