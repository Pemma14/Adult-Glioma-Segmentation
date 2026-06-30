from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def regions_to_multiclass_mask(regions: np.ndarray) -> np.ndarray:
    r"""Convert 3-channel region logits/masks into a BraTS multiclass mask.

    Channels are interpreted as WT, TC, ET. The BraTS hierarchy
    ``ET ⊂ TC ⊂ WT`` is enforced before encoding:
    ``0=background, 1=WT\TC, 2=TC\ET, 3=ET``.
    """
    wt = regions[0].astype(bool)
    tc = regions[1].astype(bool)
    et = regions[2].astype(bool)

    tc = np.logical_or(tc, et)
    wt = np.logical_or(wt, tc)

    mask = np.zeros(wt.shape, dtype=np.uint8)
    mask[wt] = 1
    mask[tc] = 2
    mask[et] = 3
    return mask


def save_prediction(
    prediction: np.ndarray,
    case_id: str,
    image_path: str | Path,
    output_dir: Path,
    save_regions: bool,
) -> dict[str, str]:
    """Save a multi-class prediction mask (and optionally per-region masks) as NIfTI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(image_path))
    affine = reference.affine

    multiclass_mask = regions_to_multiclass_mask(prediction)
    if multiclass_mask.shape != reference.shape[:3]:
        logger.warning(
            "Prediction shape %s differs from original image shape %s for %s. "
            "Saved NIfTI will be in the preprocessed space.",
            multiclass_mask.shape,
            reference.shape[:3],
            case_id,
        )

    mask_path = output_dir / f"{case_id}_pred_mask.nii.gz"
    nib.save(nib.Nifti1Image(multiclass_mask, affine), str(mask_path))

    output_paths = {"prediction_path": str(mask_path)}
    if save_regions:
        regions_path = output_dir / f"{case_id}_pred_regions.nii.gz"
        regions_4d = np.moveaxis(prediction.astype(np.uint8), 0, -1)
        nib.save(nib.Nifti1Image(regions_4d, affine), str(regions_path))
        output_paths["region_prediction_path"] = str(regions_path)
    return output_paths


def save_uncertainty_map(
    uncertainty: np.ndarray,
    case_id: str,
    image_path: str | Path,
    output_dir: Path,
) -> str:
    """Save a voxel-wise uncertainty map as a NIfTI file.

    Args:
        uncertainty: 3D array of shape (D, H, W) with float uncertainty values.
        case_id: Patient/case identifier.
        image_path: Path to the reference image used for the affine matrix.
        output_dir: Directory where the uncertainty map will be saved.

    Returns:
        Path to the saved uncertainty map.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(image_path))
    affine = reference.affine

    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    if uncertainty.shape != reference.shape[:3]:
        logger.warning(
            "Uncertainty shape %s differs from original image shape %s for %s. "
            "Saved NIfTI will be in the preprocessed space.",
            uncertainty.shape,
            reference.shape[:3],
            case_id,
        )

    uncertainty_path = output_dir / f"{case_id}_uncertainty.nii.gz"
    nib.save(nib.Nifti1Image(uncertainty, affine), str(uncertainty_path))
    return str(uncertainty_path)
