"""Command-line interface for clinical glioma segmentation inference."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

from src.glioma.inference import predict
from src.glioma.settings import get_model_version

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clinical glioma segmentation inference on a single MRI scan."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input NIfTI image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the segmentation mask and reports",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default=None,
        help="Model registry version (overrides GLIOMA__MODEL_VERSION env variable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, cpu, or a torch device string",
    )
    parser.add_argument(
        "--single_model",
        action="store_true",
        help="Use only the first fold (fold 0) instead of the full ensemble",
    )
    parser.add_argument(
        "--save_uncertainty",
        action="store_true",
        help="Save the voxel-wise uncertainty map",
    )
    parser.add_argument(
        "--save_regions",
        action="store_true",
        help="Save per-region binary masks as a 4D NIfTI",
    )
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Save PNG visualizations of selected axial slices",
    )
    parser.add_argument(
        "--n_slices",
        type=int,
        default=3,
        help="Number of axial slices to visualize",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Save a JSON report with volumes and metadata",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.model_version:
        os.environ["GLIOMA__MODEL_VERSION"] = args.model_version

    version = get_model_version()
    logger.info("Using model version: %s", version)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)

    folds = [0] if args.single_model else None
    if args.single_model:
        logger.info("Single model mode: using fold 0 only")

    result = predict(
        image_path=args.image,
        output_dir=args.output_dir,
        device=device,
        folds=folds,
        save_uncertainty=args.save_uncertainty,
        save_regions=args.save_regions,
        save_visualization=args.save_visualization,
        n_slices=args.n_slices,
    )

    if args.report:
        output_dir = Path(args.output_dir)
        report_path = output_dir / f"{result['case_id']}_report.json"
        report = {
            "case_id": result["case_id"],
            "model_version": version,
            "volumes_ml": result["volumes_ml"],
            "prediction_path": result.get("prediction_path"),
            "uncertainty_path": result.get("uncertainty_path"),
            "region_prediction_path": result.get("region_prediction_path"),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved report to %s", report_path)

    logger.info("Done. Outputs saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
