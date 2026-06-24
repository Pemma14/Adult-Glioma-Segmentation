"""Array helpers shared across the production glioma segmentation package."""

from __future__ import annotations

import numpy as np

REGION_OPTIONS = {"wt", "tc", "et"}


def multiclass_to_region(mask: np.ndarray, region: str) -> np.ndarray:
    """Convert a BraTS-style multiclass mask to a boolean region mask.

    BraTS encoding: 0=background, 1=WT\\TC, 2=TC\\ET, 3=ET.
    Regions obey the hierarchy ET ⊂ TC ⊂ WT.
    """
    region = region.lower()
    if region not in REGION_OPTIONS:
        raise ValueError(f"Unknown region: {region!r}. Use one of {REGION_OPTIONS}")

    et = mask == 3
    tc = (mask == 2) | et
    wt = (mask == 1) | tc
    return {"wt": wt, "tc": tc, "et": et}[region]
