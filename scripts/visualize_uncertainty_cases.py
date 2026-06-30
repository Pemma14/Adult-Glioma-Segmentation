"""Visualize inference predictions, errors, and uncertainty maps for selected cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def multiclass_to_region(mask: np.ndarray, region: str) -> np.ndarray:
    et = mask == 3
    tc = (mask == 2) | et
    wt = (mask == 1) | tc
    return {"wt": wt, "tc": tc, "et": et}[region.lower()]


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    s = np.asarray(slice_2d, dtype=np.float32)
    min_val = s.min()
    max_val = s.max()
    if max_val > min_val:
        s = (s - min_val) / (max_val - min_val)
    return s


def select_slices(label: np.ndarray, pred: np.ndarray, n_slices: int = 3) -> list[int]:
    foreground = label | pred
    z_counts = foreground.sum(axis=(0, 1))
    z_indices = np.where(z_counts > 0)[0]
    if len(z_indices) == 0:
        center = label.shape[2] // 2
        slice_indices = list(range(center - n_slices // 2, center + n_slices // 2 + 1))
    else:
        heaviest = int(np.argmax(z_counts))
        half = n_slices // 2
        slice_indices = list(range(heaviest - half, heaviest + half + 1))
    slice_indices = [max(0, min(s, label.shape[2] - 1)) for s in slice_indices]
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
    modality: int = 0,
) -> None:
    if image.ndim == 4:
        img_slice = normalize_slice(image[:, :, slice_idx, modality])
    else:
        img_slice = normalize_slice(image[:, :, slice_idx])
    label_slice = label[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx]
    unc_slice = uncertainty[:, :, slice_idx]

    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    region_title = region.upper()
    dice_key = f"dice_{region.lower()}"
    hd95_key = f"hd95_{region.lower()}"
    fig.suptitle(
        f"{case_id} | {region_title} | slice {slice_idx} | "
        f"mean_dice={metrics.get('mean_dice', 0):.3f} | mean_hd95={metrics.get('mean_hd95', 0):.2f}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"Image\n{case_id} ({region_title})")
    axes[0].axis("off")

    axes[1].imshow(img_slice, cmap="gray")
    axes[1].imshow(label_slice, cmap="Reds", alpha=0.5 * (label_slice > 0))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(img_slice, cmap="gray")
    axes[2].imshow(pred_slice, cmap="Greens", alpha=0.5 * (pred_slice > 0))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fp = (pred_slice == 1) & (label_slice == 0)
    fn = (pred_slice == 0) & (label_slice == 1)
    error_rgb = np.zeros((*pred_slice.shape, 3))
    error_rgb[..., 2] = fp.astype(float)  # blue
    error_rgb[..., 0] = fn.astype(float)  # yellow/red
    axes[3].imshow(img_slice, cmap="gray")
    axes[3].imshow(error_rgb, alpha=0.6 * ((fp | fn) > 0))
    axes[3].set_title("Errors: FP=blue, FN=yellow")
    axes[3].axis("off")

    im = axes[4].imshow(unc_slice, cmap="hot")
    axes[4].set_title("Uncertainty (TTA std)")
    axes[4].axis("off")
    plt.colorbar(im, ax=axes[4], fraction=0.046)

    axes[5].imshow(img_slice, cmap="gray")
    axes[5].imshow(pred_slice, cmap="Greens", alpha=0.4 * (pred_slice > 0))
    axes[5].imshow(unc_slice, cmap="hot", alpha=0.4 * (unc_slice > 0))
    axes[5].set_title("Prediction + Uncertainty")
    axes[5].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_case(
    case_row: pd.Series,
    predictions_dir: Path,
    uncertainty_dir: Path,
    output_dir: Path,
    regions: list[str],
    n_slices: int,
    modality: int = 0,
) -> None:
    case_id = case_row["case_id"]
    image_path = Path(case_row["image_path"])
    label_path = Path(case_row["label_path"])
    pred_path = predictions_dir / f"{case_id}_pred_mask.nii.gz"
    uncertainty_path = uncertainty_dir / f"{case_id}_uncertainty.nii.gz"

    if not image_path.exists():
        print(f"Image not found for {case_id}: {image_path}")
        return
    if not label_path.exists():
        print(f"Label not found for {case_id}: {label_path}")
        return
    if not pred_path.exists():
        print(f"Prediction not found for {case_id}: {pred_path}")
        return
    if not uncertainty_path.exists():
        print(f"Uncertainty not found for {case_id}: {uncertainty_path}")
        return

    image = nib.load(str(image_path)).get_fdata()
    label_multiclass = nib.load(str(label_path)).get_fdata().astype(np.uint8)
    pred_multiclass = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
    uncertainty = nib.load(str(uncertainty_path)).get_fdata().astype(np.float32)

    metrics = {
        k: float(case_row[k])
        for k in [
            "mean_dice", "dice_wt", "dice_tc", "dice_et",
            "mean_hd95", "hd95_wt", "hd95_tc", "hd95_et",
        ]
        if k in case_row
    }

    for region in regions:
        label_mask = multiclass_to_region(label_multiclass, region)
        pred_mask = multiclass_to_region(pred_multiclass, region)
        region_out_dir = output_dir / region
        region_out_dir.mkdir(parents=True, exist_ok=True)

        slice_indices = select_slices(label_mask, pred_mask, n_slices=n_slices)
        for slice_idx in slice_indices:
            out_path = region_out_dir / f"{case_id}_slice{slice_idx:03d}.png"
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
                modality=modality,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize inference uncertainty and errors.")
    parser.add_argument("--metrics_source", type=str, required=True, help="Path to inference_metrics.csv")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Directory with prediction NIfTIs")
    parser.add_argument("--uncertainty_dir", type=str, required=True, help="Directory with uncertainty NIfTIs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for PNGs")
    parser.add_argument("--max_cases", type=int, default=None, help="Maximum number of cases to visualize")
    parser.add_argument("--hd95_threshold", type=float, default=10.0, help="Select cases with mean_hd95 > threshold")
    parser.add_argument("--dice_threshold", type=float, default=0.95, help="Select cases with mean_dice < threshold")
    parser.add_argument("--regions", nargs="+", default=["wt"], choices=["wt", "tc", "et"], help="Regions to visualize")
    parser.add_argument("--n_slices", type=int, default=3, help="Number of axial slices per case/region")
    parser.add_argument("--modality", type=int, default=0, help="Image modality channel (0=FLAIR, 1=T1, 2=T1ce, 3=T2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics_source)
    predictions_dir = Path(args.predictions_dir)
    uncertainty_dir = Path(args.uncertainty_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problematic = metrics[
        (metrics["mean_hd95"] > args.hd95_threshold) | (metrics["mean_dice"] < args.dice_threshold)
    ]
    if args.max_cases is not None:
        problematic = problematic.head(args.max_cases)

    print(f"Selected {len(problematic)} problematic cases (hd95 > {args.hd95_threshold:.2f} OR dice < {args.dice_threshold:.2f})")

    for _, row in tqdm(problematic.iterrows(), desc="Visualizing"):
        visualize_case(
            case_row=row,
            predictions_dir=predictions_dir,
            uncertainty_dir=uncertainty_dir,
            output_dir=output_dir,
            regions=args.regions,
            n_slices=args.n_slices,
            modality=args.modality,
        )

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
