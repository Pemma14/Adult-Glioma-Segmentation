"""Production package for adult glioma segmentation."""

from src.glioma.settings import get_model_version, load_model_config

__all__ = ["get_model_version", "load_model_config"]
