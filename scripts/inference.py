from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
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
from scripts.utils.model import (
    load_model_weights,
    resolve_checkpoint_path,
    resolve_ensemble_paths,
    get_checkpoint_best_dice,
)
from scripts.utils.output import save_prediction, save_uncertainty_map
from scripts.utils.transforms import build_postprocess_transform, ConvertToMultiChannelMSDd
from scripts.utils.tta import get_tta_variant_count, tta_sliding_window_inference
from scripts.utils.visualization import (
    log_inference_example,
    plot_dice_summary,
    plot_hd95_summary,
)


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


def _prob_to_logit(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert probabilities back to logits for compatibility with post-processing."""
    probs = probs.clamp(min=eps, max=1.0 - eps)
    return torch.log(probs / (1.0 - probs))


def _load_ensemble_models(
    checkpoint_paths: list[Path],
    config: dict,
    device: torch.device,
) -> list[torch.nn.Module]:
    """Load all ensemble models into memory once.

    This is the default mode: models stay loaded for the whole inference loop,
    avoiding repeated state_dict loads per case.
    """
    model_name = config["model_name"]
    models: list[torch.nn.Module] = []
    for cp_path in checkpoint_paths:
        logger.info("Loading ensemble member: %s", cp_path)
        model = get_model(model_name, config).to(device)
        load_model_weights(model, cp_path, device, model_name=model_name)
        model.eval()
        models.append(model)
    return models


def ensemble_sliding_window_inference(
    inputs: torch.Tensor,
    device: torch.device,
    roi_size: tuple[int, ...],
    sw_batch_size: int,
    overlap: float,
    overlap_mode: str,
    tta_mode: str,
    use_amp: bool,
    models: list[torch.nn.Module] | None = None,
    checkpoint_paths: list[Path] | None = None,
    config: dict | None = None,
    ensemble_weights: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference with an ensemble of models and return mean logits + uncertainty.

    Accepts either a list of already-loaded ``models`` (fast, default in the
    inference script) or ``checkpoint_paths`` (loads models sequentially inside
    this call, useful for low-memory mode).

    Mean probabilities across the ensemble are converted back to logits so that
    existing post-processing transforms (sigmoid + threshold) keep working.
    Uncertainty is estimated as the average standard deviation of class
    probabilities across ensemble members (inter-model disagreement).
    """
    if models is not None and checkpoint_paths is not None:
        raise ValueError("Specify either models or checkpoint_paths, not both.")
    if models is None and checkpoint_paths is None:
        raise ValueError("Either models or checkpoint_paths must be provided.")

    unload_after = False
    if models is None:
        if config is None:
            raise ValueError("config is required when loading from checkpoint_paths.")
        models = _load_ensemble_models(checkpoint_paths, config, device)
        unload_after = True

    n_models = len(models)
    if n_models == 0:
        raise ValueError("At least one ensemble model is required.")

    if ensemble_weights is None:
        weights = torch.ones(n_models, dtype=torch.float32, device=device) / n_models
    else:
        if len(ensemble_weights) != n_models:
            raise ValueError(
                f"Number of ensemble weights ({len(ensemble_weights)}) must match "
                f"number of models ({n_models})."
            )
        weights = torch.tensor(ensemble_weights, dtype=torch.float32, device=device)
        weights = weights / weights.sum()

    all_probs: list[torch.Tensor] = []

    for model in models:
        with torch.no_grad():
            logits = tta_sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                overlap_mode=overlap_mode,
                tta_mode=tta_mode,
                use_amp=use_amp,
                return_uncertainty=False,
            )
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            all_probs.append(torch.sigmoid(logits))

    if unload_after:
        for model in models:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stacked_probs = torch.stack(all_probs, dim=0)  # (M, B, C, D, H, W)
    weights_view = weights.view(-1, 1, 1, 1, 1, 1)
    mean_probs = (stacked_probs * weights_view).sum(dim=0)  # (B, C, D, H, W)
    mean_logits = _prob_to_logit(mean_probs)

    # Inter-model disagreement averaged over classes.
    uncertainty = torch.std(stacked_probs, dim=0, unbiased=False).mean(dim=1)  # (B, D, H, W)

    return mean_logits, uncertainty


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
    clearml_task = None
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
        clearml_task = task
        clearml_logger = task.get_logger()
        logger.info("Initialized ClearML inference task: %s", task.id)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    # Resolve single vs ensemble checkpoint configuration.
    use_ensemble = (
        args.ensemble_checkpoints is not None
        or args.ensemble_folds is not None
        or args.ensemble_clearml_model_ids is not None
    )
    if use_ensemble and (args.checkpoint is not None or args.clearml_model_id is not None):
        raise ValueError(
            "Ensemble inference (--ensemble_checkpoints, --ensemble_folds or --ensemble_clearml_model_ids) "
            "is mutually exclusive with --checkpoint and --clearml_model_id."
        )

    ensemble_checkpoint_paths: list[Path] | None = None
    ensemble_weights: list[float] | None = None
    if use_ensemble:
        fold_indices = None
        if args.ensemble_folds is not None:
            fold_indices = [int(f.strip()) for f in args.ensemble_folds.split(",")]
        ensemble_checkpoint_paths = resolve_ensemble_paths(
            checkpoint_paths=args.ensemble_checkpoints,
            fold_indices=fold_indices,
            clearml_model_ids=args.ensemble_clearml_model_ids,
            model_name=config["model_name"],
            root_dir=ROOT_DIR,
            is_best=True,
        )
        logger.info(
            "Ensemble inference with %d checkpoints: %s",
            len(ensemble_checkpoint_paths),
            [str(p) for p in ensemble_checkpoint_paths],
        )

        if args.ensemble_weights is not None and args.ensemble_weight_by_dice:
            raise ValueError("--ensemble_weights and --ensemble_weight_by_dice are mutually exclusive.")

        if args.ensemble_weight_by_dice:
            raw_dices = [get_checkpoint_best_dice(p, device=device) for p in ensemble_checkpoint_paths]
            logger.info("Checkpoint best validation Dice values: %s", raw_dices)
            valid_dices = [d if d is not None and d > 0 else 0.0 for d in raw_dices]
            total = sum(valid_dices)
            if total <= 0:
                logger.warning(
                    "Could not read positive best_dice from any checkpoint. Falling back to equal weights."
                )
                ensemble_weights = None
            else:
                ensemble_weights = [d / total for d in valid_dices]
                logger.info("Ensemble weights derived from validation Dice: %s", ensemble_weights)
        elif args.ensemble_weights is not None:
            if len(args.ensemble_weights) != len(ensemble_checkpoint_paths):
                raise ValueError(
                    f"Number of --ensemble_weights ({len(args.ensemble_weights)}) must match "
                    f"number of ensemble checkpoints ({len(ensemble_checkpoint_paths)})."
                )
            ensemble_weights = list(args.ensemble_weights)
            logger.info("Using explicit ensemble weights: %s", ensemble_weights)
        else:
            ensemble_weights = None

        if args.ensemble_low_memory:
            ensemble_models: list[torch.nn.Module] | None = None
            logger.info("Ensemble low-memory mode enabled: models will be loaded per case.")
        else:
            ensemble_models = _load_ensemble_models(ensemble_checkpoint_paths, config, device)
            logger.info("Loaded %d ensemble models into memory.", len(ensemble_models))
    else:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.clearml_model_id, root_dir=ROOT_DIR)
        model = get_model(config["model_name"], config).to(device)
        load_model_weights(model, checkpoint_path, device, model_name=config["model_name"])
        model.eval()

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

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    uncertainty_dir = output_dir / "uncertainty"

    rows = []
    use_amp = device.type == "cuda" and not args.no_amp
    roi_size = tuple(config["img_size"])
    sw_batch_size = int(config.get("sw_batch_size", 1))
    tta_count = get_tta_variant_count(args.tta)

    if args.save_uncertainty and args.tta == "none" and not use_ensemble:
        logger.warning(
            "--save_uncertainty is set but TTA is disabled and ensemble is not used. "
            "The uncertainty map will be all zeros."
        )

    logger.info(
        "Inference config: overlap=%.2f, overlap_mode=%s, TTA=%s (%d variants), threshold=%.2f, postprocess=%s, save_uncertainty=%s, ensemble=%s",
        args.overlap,
        args.overlap_mode,
        args.tta,
        tta_count,
        args.threshold,
        args.postprocess,
        args.save_uncertainty,
        use_ensemble,
    )

    postprocess = build_postprocess_transform(args.postprocess, args.threshold)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            inputs = batch["image"].to(device, non_blocking=True)
            case_id = batch["case_id"][0]
            image_path = batch["image_path"][0]

            if use_ensemble:
                if args.ensemble_low_memory:
                    logits, uncertainty = ensemble_sliding_window_inference(
                        inputs=inputs,
                        device=device,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        overlap=args.overlap,
                        overlap_mode=args.overlap_mode,
                        tta_mode=args.tta,
                        use_amp=use_amp,
                        checkpoint_paths=ensemble_checkpoint_paths,
                        config=config,
                        ensemble_weights=ensemble_weights,
                    )
                else:
                    logits, uncertainty = ensemble_sliding_window_inference(
                        inputs=inputs,
                        device=device,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        overlap=args.overlap,
                        overlap_mode=args.overlap_mode,
                        tta_mode=args.tta,
                        use_amp=use_amp,
                        models=ensemble_models,
                        ensemble_weights=ensemble_weights,
                    )
                if not args.save_uncertainty:
                    uncertainty = None
            elif args.save_uncertainty:
                logits, uncertainty = tta_sliding_window_inference(
                    inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=args.overlap,
                    overlap_mode=args.overlap_mode,
                    tta_mode=args.tta,
                    use_amp=use_amp,
                    return_uncertainty=True,
                )
            else:
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
                uncertainty = None

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

            if args.save_uncertainty and uncertainty is not None:
                uncertainty_path = save_uncertainty_map(
                    uncertainty[0].cpu().numpy(),
                    case_id,
                    image_path,
                    uncertainty_dir,
                )
                row["uncertainty_path"] = uncertainty_path

            rows.append(row)

    if use_ensemble and ensemble_models is not None:
        logger.info("Unloading ensemble models from memory.")
        for model in ensemble_models:
            del model
        ensemble_models = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    results_path = output_dir / "inference_metrics.csv"
    results.to_csv(results_path, index=False)
    logger.info("Saved per-case results to %s", results_path)

    if with_labels:
        metrics = {
            metric: float(results[metric].mean())
            for metric in [
                "mean_dice", "dice_wt", "dice_tc", "dice_et",
                "mean_hd95", "hd95_wt", "hd95_tc", "hd95_et",
            ]
            if metric in results
        }
        summary = {
            "metadata": {
                "model_name": config["model_name"],
                "dataset": args.dataset,
                "fold": args.fold,
                "num_cases": len(results),
                "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
                "clearml_model_id": args.clearml_model_id,
                "ensemble": {
                    "enabled": use_ensemble,
                    "checkpoints": [str(p) for p in ensemble_checkpoint_paths] if use_ensemble else None,
                    "weights": ensemble_weights if use_ensemble else None,
                    "weight_by_dice": args.ensemble_weight_by_dice if use_ensemble else False,
                    "low_memory": args.ensemble_low_memory if use_ensemble else False,
                    "clearml_model_ids": args.ensemble_clearml_model_ids if use_ensemble else None,
                },
                "config": {
                    "img_size": config.get("img_size"),
                    "sw_batch_size": config.get("sw_batch_size"),
                    "tta": args.tta,
                    "postprocess": args.postprocess,
                    "threshold": args.threshold,
                    "overlap": args.overlap,
                    "overlap_mode": args.overlap_mode,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "metrics": metrics,
        }
        summary_path = output_dir / "inference_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Mean external metrics: %s", metrics)
        logger.info("Saved summary to %s", summary_path)

        if clearml_task is not None:
            clearml_task.upload_artifact(
                name="inference_metrics_csv",
                artifact_object=results,
                metadata={"num_cases": len(results)},
            )
            clearml_task.upload_artifact(
                name="inference_summary_json",
                artifact_object=summary,
            )
            metrics_summary_df = pd.DataFrame([metrics])
            clearml_logger.report_table(
                title="inference_summary/metrics_table",
                series="metrics",
                iteration=0,
                table_plot=metrics_summary_df,
            )

        ensemble_suffix = " (ensemble)" if use_ensemble else ""
        base_title = f"{config['model_name']}{ensemble_suffix} on {args.dataset}"
        dice_fig = plot_dice_summary(results, title=f"Dice Summary: {base_title}")
        hd95_fig = plot_hd95_summary(results, title=f"HD95 Summary: {base_title}")

        if clearml_logger is not None:
            clearml_logger.report_matplotlib_figure(
                title="inference_summary/dice",
                series="dice_summary_plot",
                iteration=0,
                figure=dice_fig,
                report_image=False,
            )
            clearml_logger.report_matplotlib_figure(
                title="inference_summary/hd95",
                series="hd95_summary_plot",
                iteration=0,
                figure=hd95_fig,
                report_image=False,
            )

        dice_plot_path = output_dir / "inference_summary_dice.png"
        hd95_plot_path = output_dir / "inference_summary_hd95.png"
        dice_fig.savefig(dice_plot_path, dpi=150, bbox_inches="tight")
        hd95_fig.savefig(hd95_plot_path, dpi=150, bbox_inches="tight")
        plt.close(dice_fig)
        plt.close(hd95_fig)
        logger.info("Saved summary plots to %s and %s", dice_plot_path, hd95_plot_path)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3D glioma segmentation inference on processed MSD/UPENN cases.")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Path to base config")
    parser.add_argument("--config", type=str, default="configs/swin_unetr.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best model checkpoint")
    parser.add_argument("--clearml_model_id", type=str, default=None, help="ClearML model ID to download; also initializes a ClearML inference task")
    parser.add_argument("--ensemble_checkpoints", nargs="+", default=None, help="List of checkpoint paths for ensemble inference (mutually exclusive with --checkpoint and --ensemble_folds)",)
    parser.add_argument("--ensemble_folds", type=str, default=None, help="Comma-separated fold indices for ensemble inference, e.g. '0,1,2,3,4'. " "Checkpoints are auto-resolved as best_model_<model>_fold<N>.pth.",)
    parser.add_argument("--ensemble_weights", type=float, nargs="+", default=None, help="Optional per-checkpoint weights for weighted ensemble averaging. " "If omitted, equal weights are used.",)
    parser.add_argument("--ensemble_weight_by_dice", action="store_true", help="Automatically weight ensemble members by their stored best validation Dice. " "Mutually exclusive with --ensemble_weights.",)
    parser.add_argument("--ensemble_low_memory", action="store_true", help="Load ensemble models sequentially for each case instead of keeping all in memory. " "Slower, but useful when GPU memory is insufficient for all models.",)
    parser.add_argument("--ensemble_clearml_model_ids", nargs="+", default=None, help="List of ClearML model IDs for ensemble inference. " "Mutually exclusive with --checkpoint, --clearml_model_id, --ensemble_checkpoints and --ensemble_folds.",)
    parser.add_argument("--clearml_debug_samples", action="store_true", help="Log per-case inference examples as debug samples in ClearML")
    parser.add_argument("--clearml_project", type=str, default="AdultGliomaSegmentation", help="ClearML project name for the inference task")
    parser.add_argument("--clearml_task_name", type=str, default=None, help="ClearML task name for the inference task (auto-generated if not set)")
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
    parser.add_argument("--overlap_mode", type=str, default="gaussian", choices=["constant", "gaussian", "blend"], help="Patch aggregation mode: constant=uniform averaging, gaussian=center-weighted, blend=gaussian+border",)
    parser.add_argument("--tta", type=str, default="none", choices=["none", "flips", "full", "rot90"], help= ("Test-time augmentation: none / flips (4 variants: original + 3 single-axis flips) / " "full (8 variants) / rot90 (4 axial 90-degree rotations)" ),)
    parser.add_argument("--postprocess", type=str, default="none", choices=["none", "largest_cc"], help=("Post-processing for region predictions: none / largest_cc " "(sigmoid + threshold + KeepLargestConnectedComponent per region)"),)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu, or a torch device string")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision on CUDA")
    parser.add_argument("--no_labels", action="store_true", help="Run inference without labels and skip metrics")
    parser.add_argument("--no_save_nifti", action="store_true", help="Do not save NIfTI predictions")
    parser.add_argument("--save_regions", action="store_true", help="Additionally save 4D WT/TC/ET binary region predictions")
    parser.add_argument("--save_uncertainty", action="store_true", help="Save a TTA-based voxel-wise uncertainty map as {case_id}_uncertainty.nii.gz",)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
