"""Reporting utilities for clinical glioma segmentation."""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np

from src.utils.array import multiclass_to_region

logger = logging.getLogger(__name__)


def compute_volumes(prediction: np.ndarray, reference_nifti: str | None = None) -> dict[str, float]:
    """Compute volumes (in ml) for WT, TC and ET regions.

    Args:
        prediction: 3D multi-class BraTS mask or 4D region mask
            of shape (3, D, H, W).
        reference_nifti: Path to the reference NIfTI. If provided, voxel
            volume is computed from its affine. Otherwise assumes 1 mm isotropic.

    Returns:
        Dictionary with volumes in ml for ``wt``, ``tc`` and ``et``.
    """
    if prediction.ndim == 4:
        regions = {
            "wt": prediction[0].astype(bool),
            "tc": prediction[1].astype(bool),
            "et": prediction[2].astype(bool),
        }
    else:
        regions = {
            "wt": multiclass_to_region(prediction, "wt"),
            "tc": multiclass_to_region(prediction, "tc"),
            "et": multiclass_to_region(prediction, "et"),
        }

    if reference_nifti:
        nii = nib.load(str(reference_nifti))
        voxel_vol = abs(np.linalg.det(nii.affine[:3, :3])) / 1000.0
    else:
        voxel_vol = 1.0 / 1000.0

    volumes = {
        region: float(mask.sum() * voxel_vol)
        for region, mask in regions.items()
    }

    logger.info("Computed volumes: %s (voxel=%.6f ml)", volumes, voxel_vol)
    return volumes
