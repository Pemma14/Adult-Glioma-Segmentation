from __future__ import annotations

import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric

REGION_NAMES = ("WT", "TC", "ET")


def compute_dice_per_region(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute per-region Dice scores and their mean."""
    scores = {}
    for index, region_name in enumerate(REGION_NAMES):
        pred_region = prediction[index].astype(bool)
        target_region = target[index].astype(bool)
        denominator = pred_region.sum() + target_region.sum()
        if denominator == 0:
            dice = 1.0
        else:
            dice = 2.0 * np.logical_and(pred_region, target_region).sum() / denominator
        scores[f"dice_{region_name.lower()}"] = float(dice)
    scores["mean_dice"] = float(np.mean([scores[f"dice_{name.lower()}"] for name in REGION_NAMES]))
    return scores


def compute_hd95_per_region(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute per-region HD95 using MONAI HausdorffDistanceMetric."""
    scores = {}
    region_values = []
    for index, region_name in enumerate(REGION_NAMES):
        pred_region = prediction[index].astype(bool)
        target_region = target[index].astype(bool)

        if pred_region.sum() == 0 and target_region.sum() == 0:
            hd95 = 0.0
        elif pred_region.sum() == 0 or target_region.sum() == 0:
            hd95 = float("nan")
        else:
            hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95)
            pred_tensor = torch.from_numpy(pred_region).unsqueeze(0).unsqueeze(0).float()
            target_tensor = torch.from_numpy(target_region).unsqueeze(0).unsqueeze(0).float()
            hd_metric(y_pred=pred_tensor, y=target_tensor)
            hd95 = float(hd_metric.aggregate().item())

        scores[f"hd95_{region_name.lower()}"] = hd95
        region_values.append(hd95)

    scores["mean_hd95"] = float(np.nanmean(region_values))
    return scores
