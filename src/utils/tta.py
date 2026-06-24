from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference


def get_tta_flip_configs(tta_mode: str) -> list[tuple[bool, bool, bool]]:
    """Return spatial flip configurations as (flip_d, flip_h, flip_w)."""
    if tta_mode == "none":
        return [(False, False, False)]
    if tta_mode == "flips":
        return [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ]
    if tta_mode == "full":
        configs = []
        for fd in (False, True):
            for fh in (False, True):
                for fw in (False, True):
                    configs.append((fd, fh, fw))
        return configs
    raise ValueError(f"Unknown TTA mode: {tta_mode}")


def get_tta_variant_count(tta_mode: str) -> int:
    """Return the number of forward passes implied by a TTA mode."""
    if tta_mode == "none":
        return 1
    if tta_mode == "flips":
        return 4
    if tta_mode == "full":
        return 8
    if tta_mode == "rot90":
        return 4
    raise ValueError(f"Unknown TTA mode: {tta_mode}")


def apply_flip_3d(tensor: torch.Tensor, flip_config: tuple[bool, bool, bool]) -> torch.Tensor:
    """Flip spatial dimensions (D, H, W) of a tensor with shape (..., D, H, W)."""
    dims = []
    for i, flip in enumerate(flip_config):
        if flip:
            dims.append(-3 + i)
    if not dims:
        return tensor
    return torch.flip(tensor, dims=dims)


def _tta_uncertainty(stacked_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute a per-voxel uncertainty map from stacked TTA logits.

    Args:
        stacked_logits: tensor of shape (V, B, C, D, H, W) where V is the
            number of TTA variants.

    Returns:
        Uncertainty map of shape (B, D, H, W): average across classes of the
        standard deviation of sigmoid probabilities across TTA variants.
    """
    probs = torch.sigmoid(stacked_logits)
    # Population std across TTA variants (unbiased=False avoids NaN when V == 1).
    std_per_class = torch.std(probs, dim=0, unbiased=False)  # (B, C, D, H, W)
    return std_per_class.mean(dim=1)  # (B, D, H, W)


def tta_sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: tuple[int, ...],
    sw_batch_size: int,
    predictor: torch.nn.Module,
    overlap: float,
    overlap_mode: str,
    tta_mode: str,
    use_amp: bool,
    return_uncertainty: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Run sliding-window inference with optional TTA and average logits.

    If ``return_uncertainty`` is True, also returns a voxel-wise uncertainty map
    estimated as the average standard deviation of class probabilities across
    TTA variants.
    """
    all_logits: list[torch.Tensor] = []

    if tta_mode == "rot90":
        for k in range(4):
            x_rot = torch.rot90(inputs, k, dims=(-2, -1))
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = sliding_window_inference(
                    x_rot,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=predictor,
                    overlap=overlap,
                    mode=overlap_mode,
                    sigma_scale=0.125,
                )
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            logits = torch.rot90(logits, -k, dims=(-2, -1))
            all_logits.append(logits)
    else:
        flip_configs = get_tta_flip_configs(tta_mode)
        for flip_config in flip_configs:
            flipped_inputs = apply_flip_3d(inputs, flip_config)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = sliding_window_inference(
                    flipped_inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=predictor,
                    overlap=overlap,
                    mode=overlap_mode,
                    sigma_scale=0.125,
                )
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            # Reverse flip to align with original orientation
            logits = apply_flip_3d(logits, flip_config)
            all_logits.append(logits)

    stacked_logits = torch.stack(all_logits, dim=0)  # (V, B, C, D, H, W)
    mean_logits = stacked_logits.mean(dim=0)  # (B, C, D, H, W)

    if return_uncertainty:
        uncertainty = _tta_uncertainty(stacked_logits)
        return mean_logits, uncertainty

    return mean_logits
