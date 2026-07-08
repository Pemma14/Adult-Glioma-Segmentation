"""Visualization utilities for clinical glioma segmentation results."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.array import multiclass_to_region

logger = logging.getLogger(__name__)

REGION_COLORS = {
    "wt": (0.0, 0.8, 0.0),   # green
    "tc": (1.0, 0.0, 0.0),   # red
    "et": (1.0, 1.0, 0.0),   # yellow
}


def _normalize_slice(slice_: np.ndarray) -> np.ndarray:
    """Normalize a 2D image slice to [0, 1] using 1st-99th percentiles."""
    lo, hi = np.percentile(slice_, [1, 99])
    if hi - lo < 1e-6:
        return np.zeros_like(slice_, dtype=np.float32)
    return np.clip((slice_.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _select_slices(mask: np.ndarray, n_slices: int = 3) -> list[int]:
    """Pick axial slices around the center of the foreground mask."""
    n_z = mask.shape[2]
    z_counts = mask.sum(axis=(0, 1))
    z_indices = np.where(z_counts > 0)[0]
    if len(z_indices) == 0:
        center = n_z // 2
        return list(range(center - n_slices // 2, center + n_slices // 2 + 1))

    heaviest = int(np.argmax(z_counts))
    half = n_slices // 2
    slice_indices = list(range(heaviest - half, heaviest + half + 1))
    slice_indices = [max(0, min(s, n_z - 1)) for s in slice_indices]
    return sorted(set(slice_indices))


def create_case_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray | None,
    case_id: str,
    slice_idx: int,
    modality: int = 0,
) -> plt.Figure:
    """Create a multi-panel figure for a single axial slice.

    Args:
        image: Input image of shape (H, W, D, C) or (H, W, D).
        prediction: 3D multi-class BraTS mask of shape (H, W, D).
        uncertainty: Optional 3D uncertainty map of shape (H, W, D).
        case_id: Patient/case identifier.
        slice_idx: Axial slice index.
        modality: Image modality channel to display (0=FLAIR, 1=T1, 2=T1ce, 3=T2).

    Returns:
        Matplotlib figure.
    """
    if image.ndim == 4:
        img_slice = _normalize_slice(image[:, :, slice_idx, modality])
    else:
        img_slice = _normalize_slice(image[:, :, slice_idx])

    pred_slice = prediction[:, :, slice_idx]
    unc_slice = uncertainty[:, :, slice_idx] if uncertainty is not None else None

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Image
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"Image\n{case_id} (slice {slice_idx})")
    axes[0].axis("off")

    # 2. Prediction overlay
    axes[1].imshow(img_slice, cmap="gray")
    for region, color in REGION_COLORS.items():
        mask = multiclass_to_region(pred_slice, region).astype(float)
        if mask.any():
            rgba = np.zeros((*mask.shape, 4))
            rgba[..., :3] = color
            rgba[..., 3] = mask * 0.5
            axes[1].imshow(rgba)
    axes[1].set_title("Prediction overlay\nWT=green, TC=red, ET=yellow")
    axes[1].axis("off")

    # 3. Segmentation mask only
    seg_rgb = np.zeros((*pred_slice.shape, 3))
    for region, color in REGION_COLORS.items():
        mask = multiclass_to_region(pred_slice, region)
        seg_rgb[mask] = color
    axes[2].imshow(seg_rgb)
    axes[2].set_title("Segmentation mask")
    axes[2].axis("off")

    # 4. Uncertainty heatmap
    if unc_slice is not None:
        im = axes[3].imshow(unc_slice, cmap="hot")
        axes[3].set_title("Uncertainty")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    else:
        axes[3].text(0.5, 0.5, "Not computed", ha="center", va="center")
        axes[3].set_title("Uncertainty")
        axes[3].axis("off")

    plt.tight_layout()
    return fig


def save_prediction_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray | None,
    case_id: str,
    output_dir: Path,
    modality: int = 0,
    n_slices: int = 3,
) -> list[Path]:
    """Save PNG visualizations for selected axial slices.

    Returns:
        List of saved PNG paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    foreground_mask = (prediction > 0).astype(bool)
    slice_indices = _select_slices(foreground_mask, n_slices=n_slices)

    saved_paths: list[Path] = []
    for slice_idx in slice_indices:
        fig = create_case_visualization(
            image=image,
            prediction=prediction,
            uncertainty=uncertainty,
            case_id=case_id,
            slice_idx=slice_idx,
            modality=modality,
        )
        out_path = output_dir / f"{case_id}_slice{slice_idx:03d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
        logger.info("Saved visualization to %s", out_path)

    return saved_paths
