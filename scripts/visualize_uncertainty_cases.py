from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

logger = logging.getLogger(__name__)


# Custom colormaps for error overlay
FP_COLOR = np.array([0.2, 0.4, 1.0, 1.0])   # blue
FN_COLOR = np.array([1.0, 0.9, 0.2, 1.0])   # yellow
CMAP_FP = ListedColormap([FP_COLOR])
CMAP_FN = ListedColormap([FN_COLOR])


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def load_metrics(metrics_source: str | Path) -> pd.DataFrame:
    """Load metrics from a local CSV or a ClearML task ID."""
    metrics_source = str(metrics_source)
    csv_path = _resolve_path(metrics_source)
    if csv_path.exists():
        logger.info("Loading metrics from %s", csv_path)
        return pd.read_csv(csv_path)

    # Try interpreting the string as a ClearML task ID.
    try:
        from clearml import Task

        logger.info("Fetching metrics from ClearML task %s", metrics_source)
        task = Task.get_task(task_id=metrics_source)
        csv_path = task.artifacts["inference_metrics_csv"].get_local_copy()
        return pd.read_csv(csv_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load metrics from {metrics_source!r} as file or ClearML task: {e}"
        )


def multiclass_to_region(mask: np.ndarray, region: str) -> np.ndarray:
    r"""
    Convert a BraTS-style multiclass mask (0=bg, 1=WT\\TC, 2=TC\\ET, 3=ET)
    to a boolean mask for one region (WT, TC, ET).
    """
    et = mask == 3
    tc = (mask == 2) | et
    wt = (mask == 1) | tc
    return {"wt": wt, "tc": tc, "et": et}[region.lower()]


def normalize_slice(slice_: np.ndarray) -> np.ndarray:
    """Normalize image slice for display using 1st-99th percentile."""
    lo, hi = np.percentile(slice_, [1, 99])
    if hi - lo < 1e-6:
        return np.zeros_like(slice_, dtype=np.float32)
    return np.clip((slice_.astype(np.float32) - lo) / (hi - lo), 0, 1)


def select_slices(
    label: np.ndarray,
    pred: np.ndarray,
    n_slices: int = 3,
) -> list[int]:
    """
    Pick axial slices around the center of the foreground region.
    If no foreground is present, picks the center of the volume.
    """
    foreground = label | pred
    z_indices = np.where(foreground.any(axis=(1, 2)))[0]
    if len(z_indices) == 0:
        center = label.shape[0] // 2
    else:
        center = int(np.median(z_indices))

    half = n_slices // 2
    slice_indices = list(range(center - half, center + half + 1))
    slice_indices = [max(0, min(s, label.shape[0] - 1)) for s in slice_indices]
    return sorted(set(slice_indices))


def plot_case_overlay(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    case_id: str,
    region: str,
    slice_idx: int,
    metrics: dict[str, float],
    out_path: Path,
) -> None:
    """
    Create a 6-panel figure: image, GT, prediction, errors, uncertainty, pred+uncertainty.
    """
    img_slice = normalize_slice(image[0, slice_idx])
    label_slice = label[slice_idx]
    pred_slice = pred[slice_idx]
    unc_slice = uncertainty[slice_idx]

    fp = pred_slice & ~label_slice
    fn = label_slice & ~pred_slice

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    # 1. Image
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"Image\n{case_id} ({region.upper()})")
    axes[0].axis("off")

    # 2. Ground truth
    axes[1].imshow(img_slice, cmap="gray")
    axes[1].imshow(label_slice, cmap="Reds", alpha=0.5 * label_slice)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # 3. Prediction
    axes[2].imshow(img_slice, cmap="gray")
    axes[2].imshow(pred_slice, cmap="Greens", alpha=0.5 * pred_slice)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # 4. Errors (FP blue, FN yellow)
    axes[3].imshow(img_slice, cmap="gray")
    axes[3].imshow(fp, cmap=CMAP_FP, alpha=0.6 * fp)
    axes[3].imshow(fn, cmap=CMAP_FN, alpha=0.6 * fn)
    axes[3].set_title("Errors: FP=blue, FN=yellow")
    axes[3].axis("off")

    # 5. Uncertainty heatmap
    im = axes[4].imshow(unc_slice, cmap="hot")
    axes[4].set_title("Uncertainty (TTA std)")
    axes[4].axis("off")
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    # 6. Prediction + uncertainty overlay
    axes[5].imshow(img_slice, cmap="gray")
    axes[5].imshow(pred_slice, cmap="Greens", alpha=0.3 * pred_slice)
    im2 = axes[5].imshow(unc_slice, cmap="hot", alpha=0.5)
    axes[5].set_title("Prediction + Uncertainty")
    axes[5].axis("off")
    plt.colorbar(im2, ax=axes[5], fraction=0.046, pad=0.04)

    title = (
        f"{case_id} | {region.upper()} | slice {slice_idx} | "
        f"mean_dice={metrics.get('mean_dice', -1):.3f} | "
        f"mean_hd95={metrics.get('mean_hd95', -1):.2f}"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_case(
    case_row: pd.Series,
    predictions_dir: Path,
    uncertainty_dir: Path,
    output_dir: Path,
    regions: list[str],
    n_slices: int,
) -> None:
    case_id = str(case_row["case_id"])
    image_path = _resolve_path(case_row["image_path"])
    label_path = _resolve_path(case_row["label_path"])
    pred_path = predictions_dir / f"{case_id}_pred_mask.nii.gz"
    uncertainty_path = uncertainty_dir / f"{case_id}_uncertainty.nii.gz"

    if not image_path.exists():
        logger.warning("Image not found for %s: %s", case_id, image_path)
        return
    if not label_path.exists():
        logger.warning("Label not found for %s: %s", case_id, label_path)
        return
    if not pred_path.exists():
        logger.warning("Prediction not found for %s: %s", case_id, pred_path)
        return
    if not uncertainty_path.exists():
        logger.warning("Uncertainty map not found for %s: %s", case_id, uncertainty_path)
        return

    logger.info("Visualizing %s", case_id)

    image = nib.load(str(image_path)).get_fdata()
    label_multiclass = nib.load(str(label_path)).get_fdata().astype(np.uint8)
    pred_multiclass = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
    uncertainty = nib.load(str(uncertainty_path)).get_fdata().astype(np.float32)

    # Ensure image is (C, D, H, W)
    if image.ndim == 3:
        image = image[np.newaxis, ...]

    metrics = {
        "mean_dice": case_row.get("mean_dice", -1),
        "mean_hd95": case_row.get("mean_hd95", -1),
    }

    for region in regions:
        label_mask = multiclass_to_region(label_multiclass, region)
        pred_mask = multiclass_to_region(pred_multiclass, region)
        slice_indices = select_slices(label_mask, pred_mask, n_slices=n_slices)

        for slice_idx in slice_indices:
            out_path = output_dir / region / f"{case_id}_slice{slice_idx:03d}.png"
            plot_case_overlay(
                image=image,
                label=label_mask,
                pred=pred_mask,
                uncertainty=uncertainty,
                case_id=case_id,
                region=region,
                slice_idx=slice_idx,
                metrics=metrics,
                out_path=out_path,
            )


def select_problematic_cases(
    df: pd.DataFrame,
    max_cases: int | None,
    hd95_threshold: float | None,
    dice_threshold: float | None,
) -> pd.DataFrame:
    """Select the worst cases by HD95 and/or low Dice."""
    if hd95_threshold is not None:
        df = df[df["mean_hd95"] > hd95_threshold]
    if dice_threshold is not None:
        df = df[df["mean_dice"] < dice_threshold]

    if df.empty:
        raise ValueError("No cases matched the given criteria.")

    # Sort by worst mean_hd95, then by worst mean_dice
    df = df.sort_values(["mean_hd95", "mean_dice"], ascending=[False, True])
    if max_cases is not None:
        df = df.head(max_cases)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PNG overlays for problematic inference cases with uncertainty maps."
    )
    parser.add_argument(
        "--metrics_source",
        type=str,
        required=True,
        help="Path to inference_metrics.csv or ClearML task ID",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="results/inference/predictions",
        help="Directory with *_pred_mask.nii.gz predictions",
    )
    parser.add_argument(
        "--uncertainty_dir",
        type=str,
        default="results/inference/uncertainty",
        help="Directory with *_uncertainty.nii.gz maps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/uncertainty_visualizations",
        help="Directory to save PNG overlays",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=20,
        help="Maximum number of problematic cases to visualize",
    )
    parser.add_argument(
        "--hd95_threshold",
        type=float,
        default=10.0,
        help="Select cases with mean_hd95 above this threshold",
    )
    parser.add_argument(
        "--dice_threshold",
        type=float,
        default=0.75,
        help="Select cases with mean_dice below this threshold",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["wt", "tc", "et"],
        choices=["wt", "tc", "et"],
        help="Regions to visualize",
    )
    parser.add_argument(
        "--n_slices",
        type=int,
        default=3,
        help="Number of axial slices to visualize per case/region",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = load_metrics(args.metrics_source)
    required = {"case_id", "image_path", "label_path", "mean_dice", "mean_hd95"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {missing}")

    problematic = select_problematic_cases(
        df,
        max_cases=args.max_cases,
        hd95_threshold=args.hd95_threshold,
        dice_threshold=args.dice_threshold,
    )
    logger.info(
        "Selected %d problematic cases (hd95 > %.2f OR dice < %.2f)",
        len(problematic),
        args.hd95_threshold,
        args.dice_threshold,
    )

    predictions_dir = _resolve_path(args.predictions_dir)
    uncertainty_dir = _resolve_path(args.uncertainty_dir)
    output_dir = _resolve_path(args.output_dir)

    for _, row in problematic.iterrows():
        visualize_case(
            case_row=row,
            predictions_dir=predictions_dir,
            uncertainty_dir=uncertainty_dir,
            output_dir=output_dir,
            regions=args.regions,
            n_slices=args.n_slices,
        )

    logger.info("Saved visualizations to %s", output_dir)


if __name__ == "__main__":
    main()
