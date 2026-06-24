"""Shared helpers for inference result analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


EXPERIMENTS_DIR = Path(__file__).resolve()
while EXPERIMENTS_DIR != EXPERIMENTS_DIR.parent and not (EXPERIMENTS_DIR / "pyproject.toml").exists():
    EXPERIMENTS_DIR = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the experiments root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return EXPERIMENTS_DIR / path


def load_metrics(metrics_source: str | Path) -> pd.DataFrame:
    """Load metrics from a CSV or JSON file."""
    path = resolve_path(metrics_source)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def load_nifti(path: str | Path, dtype: Any = None) -> np.ndarray:
    """Load a NIfTI file and return its data array."""
    data = nib.load(str(path)).get_fdata()
    if dtype is not None:
        data = data.astype(dtype)
    return data


def multiclass_to_region(mask: np.ndarray, region: str | int) -> np.ndarray:
    """Extract a binary region mask from a BraTS multiclass mask.

    Multiclass labels: 1 = WT, 2 = TC, 3 = ET.
    """
    mapping = {"wt": 1, "tc": 2, "et": 3}
    if isinstance(region, str):
        value = mapping.get(region.lower())
        if value is None:
            raise ValueError(f"Unknown region '{region}'. Expected one of {list(mapping)}")
        region = value
    return (mask == region).astype(np.uint8)


def select_cases(
    df: pd.DataFrame,
    hd95_threshold: float,
    dice_threshold: float,
    max_cases: int | None,
    sort_by: str,
    mode: str,
) -> pd.DataFrame:
    """Select cases matching the HD95 / Dice criteria."""
    mode = mode.lower()
    if mode == "or":
        mask = (df["mean_hd95"] > hd95_threshold) | (df["mean_dice"] < dice_threshold)
    elif mode == "and":
        mask = (df["mean_hd95"] > hd95_threshold) & (df["mean_dice"] < dice_threshold)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'or' or 'and'.")

    selected = df[mask].copy()

    sort_by = sort_by.lower()
    if sort_by == "hd95":
        selected = selected.sort_values("mean_hd95", ascending=False)
    elif sort_by == "dice":
        selected = selected.sort_values("mean_dice", ascending=True)
    elif sort_by == "badness":
        if "badness_score" not in selected.columns:
            raise ValueError("sort_by='badness' requires a 'badness_score' column")
        selected = selected.sort_values("badness_score", ascending=False)
    else:
        raise ValueError(f"Unknown sort_by '{sort_by}'. Expected 'hd95', 'dice', or 'badness'.")

    if max_cases is not None:
        selected = selected.head(max_cases)

    return selected
