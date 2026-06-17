from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; figures are saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    SpatialPadd,
    Spacingd,
    ToTensord,
)
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models import get_model
from scripts.utils.config_schema import validate_config, get_required_inference_keys
from scripts.utils.metrics import compute_dice_per_region, compute_hd95_per_region
from scripts.utils.model import load_model_weights, resolve_checkpoint_path
from scripts.utils.output import save_prediction
from scripts.utils.transforms import build_postprocess_transform, ConvertToMultiChannelMSDd
from scripts.utils.tta import get_tta_variant_count, tta_sliding_window_inference
from scripts.utils.visualization import log_inference_example, plot_inference_summary


logger = logging.getLogger(__name__)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_config(config_path: str | Path, base_config_path: str | Path) -> dict:
    base_path = resolve_project_path(base_config_path)
    spec_path = resolve_project_path(config_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not spec_path.exists():
        raise FileNotFoundError(f"Model config not found: {spec_path}")

    with open(base_path, "r") as f:
        config = yaml.safe_load(f) or {}
    with open(spec_path, "r") as f:
        config.update(yaml.safe_load(f) or {})

    required_keys = get_required_inference_keys(config["model_name"])
    validate_config(config, required_keys, context=f"конфиге {config_path}")
    return config


def get_inference_transforms(config: dict, with_labels: bool) -> Compose:
    keys = ["image", "label"] if with_labels else ["image"]
    spacing_mode = ("bilinear", "nearest") if with_labels else ("bilinear",)

    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    ]
    if with_labels:
        transforms.append(EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"))

    transforms.extend(
        [
            Orientationd(keys=keys, axcodes="RAS", labels=None),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=spacing_mode),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    if with_labels:
        transforms.append(ConvertToMultiChannelMSDd(keys="label"))

    transforms.extend(
        [
            SpatialPadd(keys=keys, spatial_size=config["img_size"]),
            ToTensord(keys=keys),
        ]
    )
    return Compose(transforms)


def build_cases(args: argparse.Namespace) -> tuple[list[dict], bool]:
    metadata_path = resolve_project_path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if "dataset" not in df.columns:
        raise ValueError(f"Column 'dataset' not found in {metadata_path}")

    df = df[df["dataset"] == args.dataset].copy()
    if df.empty:
        raise ValueError(f"No cases found for dataset={args.dataset!r} in {metadata_path}")

    if args.fold is not None:
        if "fold" not in df.columns:
            raise ValueError("--fold was passed, but metadata does not contain a 'fold' column")
        df = df[df["fold"] == args.fold].copy()
        if df.empty:
            raise ValueError(f"No cases found for dataset={args.dataset!r}, fold={args.fold}")

    if args.case_ids:
        requested_ids = set(args.case_ids)
        df = df[df["patient_id"].astype(str).isin(requested_ids)].copy()
        missing_ids = sorted(requested_ids - set(df["patient_id"].astype(str)))
        if missing_ids:
            logger.warning("Some requested case_ids were not found: %s", ", ".join(missing_ids))
        if df.empty:
            raise ValueError("None of the requested case_ids were found")

    df = df.sort_values("patient_id")
    if args.max_cases is not None:
        df = df.head(args.max_cases)

    data_dir = resolve_project_path(args.data_dir)
    cases = []
    with_labels = not args.no_labels
    for _, row in df.iterrows():
        image_path = data_dir / row["dataset"] / row["image_path"]
        case = {
            "image": str(image_path),
            "image_path": str(image_path),
            "case_id": str(row["patient_id"]),
        }

        if with_labels:
            if "label_path" not in row or pd.isna(row["label_path"]):
                logger.warning("Label path is missing for %s. Metrics will be disabled.", row["patient_id"])
                with_labels = False
            else:
                label_path = data_dir / row["dataset"] / row["label_path"]
                if label_path.exists():
                    case["label"] = str(label_path)
                    case["label_path"] = str(label_path)
                else:
                    logger.warning("Label not found for %s: %s. Metrics will be disabled.", row["patient_id"], label_path)
                    with_labels = False
        cases.append(case)

    return cases, with_labels


def run_inference(args: argparse.Namespace) -> pd.DataFrame:
    config = load_config(args.config, args.base_config)
    if args.sw_batch_size is not None:
        config["sw_batch_size"] = args.sw_batch_size

    clearml_logger = None
    if args.clearml_model_id is not None or args.clearml_debug_samples:
        from clearml import Task
        task_name = args.clearml_task_name or f"inference_{config['model_name']}_{args.dataset}"
        task = Task.init(
            project_name=args.clearml_project,
            task_name=task_name,
            task_type=Task.TaskTypes.inference,
        )
        task.connect(config)
        if args.clearml_model_id is not None:
            task.set_parameter("inference_clearml_model_id", args.clearml_model_id)
        clearml_logger = task.get_logger()
        logger.info("Initialized ClearML inference task: %s", task.id)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    cases, with_labels = build_cases(args)
    logger.info("Prepared %d cases from %s. Metrics enabled: %s", len(cases), args.dataset, with_labels)

    transforms = get_inference_transforms(config, with_labels=with_labels)
    dataset = Dataset(data=cases, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.clearml_model_id, root_dir=ROOT_DIR)
    model = get_model(config["model_name"], config).to(device)
    load_model_weights(model, checkpoint_path, device)
    model.eval()

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"

    rows = []
    use_amp = device.type == "cuda" and not args.no_amp
    roi_size = tuple(config["img_size"])
    sw_batch_size = int(config.get("sw_batch_size", 1))
    tta_count = get_tta_variant_count(args.tta)

    logger.info(
        "Inference config: overlap=%.2f, overlap_mode=%s, TTA=%s (%d variants), threshold=%.2f, postprocess=%s",
        args.overlap,
        args.overlap_mode,
        args.tta,
        tta_count,
        args.threshold,
        args.postprocess,
    )

    postprocess = build_postprocess_transform(args.postprocess, args.threshold)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            inputs = batch["image"].to(device, non_blocking=True)
            case_id = batch["case_id"][0]
            image_path = batch["image_path"][0]

            logits = tta_sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=args.overlap,
                overlap_mode=args.overlap_mode,
                tta_mode=args.tta,
                use_amp=use_amp,
            )

            if postprocess is not None:
                prediction = postprocess(logits)[0].cpu().numpy().astype(np.uint8)
            else:
                prediction = (torch.sigmoid(logits)[0] > args.threshold).cpu().numpy().astype(np.uint8)

            row = {"case_id": case_id, "image_path": image_path}
            target = None
            if with_labels:
                target = batch["label"][0].cpu().numpy().astype(np.uint8)
                row.update(compute_dice_per_region(prediction, target))
                row.update(compute_hd95_per_region(prediction, target))
                row["label_path"] = batch["label_path"][0]

            if clearml_logger is not None and args.clearml_debug_samples:
                log_inference_example(
                    image=batch["image"][0],
                    prediction=prediction,
                    target=target,
                    case_id=case_id,
                    clearml_logger=clearml_logger,
                )

            if not args.no_save_nifti:
                row.update(save_prediction(prediction, case_id, image_path, predictions_dir, args.save_regions))

            rows.append(row)

    results = pd.DataFrame(rows)
    results_path = output_dir / "inference_metrics.csv"
    results.to_csv(results_path, index=False)
    logger.info("Saved per-case results to %s", results_path)

    if with_labels:
        summary = {
            metric: float(results[metric].mean())
            for metric in [
                "mean_dice", "dice_wt", "dice_tc", "dice_et",
                "mean_hd95", "hd95_wt", "hd95_tc", "hd95_et",
            ]
            if metric in results
        }
        summary_path = output_dir / "inference_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Mean external metrics: %s", summary)
        logger.info("Saved summary to %s", summary_path)

        if clearml_logger is not None:
            for metric_name, metric_value in summary.items():
                clearml_logger.report_scalar(
                    title="Inference Metrics",
                    series=metric_name,
                    value=metric_value,
                    iteration=0,
                )

        summary_fig = plot_inference_summary(
            results,
            title=f"Inference Summary: {config['model_name']} on {args.dataset}",
        )

        if clearml_logger is not None:
            clearml_logger.report_matplotlib_figure(
                title="inference_summary",
                series="summary_plot",
                iteration=0,
                figure=summary_fig,
                report_image=False,
            )

        summary_plot_path = output_dir / "inference_summary.png"
        summary_fig.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
        plt.close(summary_fig)
        logger.info("Saved summary plot to %s", summary_plot_path)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3D glioma segmentation inference on processed MSD/UPENN cases.")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Path to base config")
    parser.add_argument("--config", type=str, default="configs/swin_unetr.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best model checkpoint")
    parser.add_argument("--clearml_model_id", type=str, default=None,
                        help="ClearML model ID to download; also initializes a ClearML inference task")
    parser.add_argument("--clearml_debug_samples", action="store_true",
                        help="Log per-case inference examples as debug samples in ClearML")
    parser.add_argument("--clearml_project", type=str, default="AdultGliomaSegmentation",
                        help="ClearML project name for the inference task")
    parser.add_argument("--clearml_task_name", type=str, default=None,
                        help="ClearML task name for the inference task (auto-generated if not set)")
    parser.add_argument("--metadata", type=str, default="data/processed/metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data root")
    parser.add_argument("--dataset", type=str, default="UPENN-GBM", help="Dataset name from metadata.csv")
    parser.add_argument("--output_dir", type=str, default="results/inference", help="Directory for metrics and predictions")
    parser.add_argument("--fold", type=int, default=None, help="Optional fold filter, useful for MSD validation cases")
    parser.add_argument("--case_ids", nargs="*", default=None, help="Optional list of patient IDs to process")
    parser.add_argument("--max_cases", type=int, default=None, help="Optional limit for smoke tests, e.g. 5 or 20")
    parser.add_argument("--sw_batch_size", type=int, default=None, help="Override sliding-window batch size; use 1 if CUDA OOM")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for region masks")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap")
    parser.add_argument(
        "--overlap_mode",
        type=str,
        default="gaussian",
        choices=["constant", "gaussian", "blend"],
        help="Patch aggregation mode: constant=uniform averaging, gaussian=center-weighted, blend=gaussian+border",
    )
    parser.add_argument(
        "--tta",
        type=str,
        default="none",
        choices=["none", "flips", "full", "rot90"],
        help=(
            "Test-time augmentation: none / flips (4 variants: original + 3 single-axis flips) / "
            "full (8 variants) / rot90 (4 axial 90-degree rotations via TestTimeRotation wrapper)"
        ),
    )
    parser.add_argument(
        "--postprocess",
        type=str,
        default="none",
        choices=["none", "largest_cc"],
        help=(
            "Post-processing for region predictions: none / largest_cc "
            "(sigmoid + threshold + KeepLargestConnectedComponent per region)"
        ),
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu, or a torch device string")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision on CUDA")
    parser.add_argument("--no_labels", action="store_true", help="Run inference without labels and skip metrics")
    parser.add_argument("--no_save_nifti", action="store_true", help="Do not save NIfTI predictions")
    parser.add_argument("--save_regions", action="store_true", help="Additionally save 4D WT/TC/ET binary region predictions")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
