"""Output saving utilities for clinical glioma segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def regions_to_multiclass_mask(regions: np.ndarray) -> np.ndarray:
    """Convert a 3-channel region mask to a BraTS multiclass mask.

    Channels are interpreted as WT, TC, ET. The hierarchy ET ⊂ TC ⊂ WT
    is enforced: ``0=bg, 1=WT\\TC, 2=TC\\ET, 3=ET``.
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
    save_regions: bool = False,
) -> dict[str, str]:
    """Save the multi-class prediction mask as a NIfTI file.

    Args:
        prediction: 3D multi-class mask or 4D region mask of shape (3, D, H, W).
        case_id: Patient/case identifier used in the output filename.
        image_path: Reference image used for the affine matrix.
        output_dir: Directory where the mask will be saved.
        save_regions: If True, also save a 4D NIfTI with per-region channels.

    Returns:
        Dictionary with saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = nib.load(str(image_path))
    affine = reference.affine

    if prediction.ndim == 4:
        multiclass_mask = regions_to_multiclass_mask(prediction)
    else:
        multiclass_mask = prediction.astype(np.uint8)

    if multiclass_mask.shape != reference.shape[:3]:
        logger.warning(
            "Prediction shape %s differs from reference image shape %s for %s. "
            "Saved NIfTI will be in the model's processing space.",
            multiclass_mask.shape,
            reference.shape[:3],
            case_id,
        )

    mask_path = output_dir / f"{case_id}_pred_mask.nii.gz"
    nib.save(nib.Nifti1Image(multiclass_mask, affine), str(mask_path))

    result = {"prediction_path": str(mask_path)}

    if save_regions and prediction.ndim == 4:
        regions_path = output_dir / f"{case_id}_pred_regions.nii.gz"
        regions_4d = np.moveaxis(prediction.astype(np.uint8), 0, -1)
        nib.save(nib.Nifti1Image(regions_4d, affine), str(regions_path))
        result["region_prediction_path"] = str(regions_path)

    logger.info("Saved prediction to %s", mask_path)
    return result


def save_rgb_mask(
    prediction: np.ndarray,
    case_id: str,
    image_path: str | Path,
    output_dir: Path,
) -> str:
    """Save the multi-class prediction as an RGB NIfTI for color viewer overlays.

    Colors match the PNG preview:
        0=background (transparent), 1=WT\\TC (edema, green),
        2=TC\\ET (core, red), 3=ET (enhancing, yellow).

    Args:
        prediction: 3D multi-class mask or 4D region mask of shape (3, D, H, W).
        case_id: Patient/case identifier used in the output filename.
        image_path: Reference image used for the affine matrix.
        output_dir: Directory where the RGB mask will be saved.

    Returns:
        Path to the saved RGB mask.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = nib.load(str(image_path))
    affine = reference.affine

    if prediction.ndim == 4:
        multiclass_mask = regions_to_multiclass_mask(prediction)
    else:
        multiclass_mask = prediction.astype(np.uint8)

    # Map BraTS labels to RGB colors (uint8, 0-255).
    colors = {
        0: (0, 0, 0),
        1: (0, 204, 0),     # green: WT\TC
        2: (255, 0, 0),     # red: TC\ET
        3: (255, 255, 0),   # yellow: ET
    }
    rgb = np.zeros((*multiclass_mask.shape, 3), dtype=np.uint8)
    for label, (r, g, b) in colors.items():
        mask = multiclass_mask == label
        rgb[mask, 0] = r
        rgb[mask, 1] = g
        rgb[mask, 2] = b

    rgb_path = output_dir / f"{case_id}_pred_mask_rgb.nii.gz"
    nib.save(nib.Nifti1Image(rgb, affine), str(rgb_path))
    logger.info("Saved RGB mask to %s", rgb_path)
    return str(rgb_path)


def save_uncertainty_map(
    uncertainty: np.ndarray,
    case_id: str,
    image_path: str | Path,
    output_dir: Path,
) -> str:
    """Save a voxel-wise uncertainty map as a NIfTI file.

    Args:
        uncertainty: 3D array of shape (D, H, W).
        case_id: Patient/case identifier.
        image_path: Reference image used for the affine matrix.
        output_dir: Directory where the map will be saved.

    Returns:
        Path to the saved uncertainty map.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = nib.load(str(image_path))
    affine = reference.affine

    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    if uncertainty.shape != reference.shape[:3]:
        logger.warning(
            "Uncertainty shape %s differs from reference image shape %s for %s. "
            "Saved NIfTI will be in the model's processing space.",
            uncertainty.shape,
            reference.shape[:3],
            case_id,
        )

    uncertainty_path = output_dir / f"{case_id}_uncertainty.nii.gz"
    nib.save(nib.Nifti1Image(uncertainty, affine), str(uncertainty_path))
    logger.info("Saved uncertainty map to %s", uncertainty_path)
    return str(uncertainty_path)
