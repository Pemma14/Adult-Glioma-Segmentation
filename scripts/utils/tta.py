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


class TestTimeRotation:
    """Wrap a model to perform inference over 90-degree rotations and average logits."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_rotations: int = 4,
        spatial_dims: int = 3,
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.num_rotations = num_rotations
        self.spatial_dims = spatial_dims
        self.batch_size = batch_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for k in range(self.num_rotations):
            x_rot = torch.rot90(x, k, dims=(-2, -1))
            out = self.model(x_rot)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = torch.rot90(out, -k, dims=(-2, -1))
            outputs.append(out)
        return torch.stack(outputs, dim=0).mean(dim=0)


def tta_sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: tuple[int, ...],
    sw_batch_size: int,
    predictor: torch.nn.Module,
    overlap: float,
    overlap_mode: str,
    tta_mode: str,
    use_amp: bool,
) -> torch.Tensor:
    """Run sliding-window inference with optional TTA and average logits."""
    if tta_mode == "rot90":
        tta_model = TestTimeRotation(
            model=predictor,
            num_rotations=4,
            spatial_dims=3,
            batch_size=sw_batch_size,
        )
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=tta_model,
                overlap=overlap,
                mode=overlap_mode,
                sigma_scale=0.125,
            )
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return logits

    flip_configs = get_tta_flip_configs(tta_mode)
    accumulated_logits: torch.Tensor | None = None

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

        if accumulated_logits is None:
            accumulated_logits = logits
        else:
            accumulated_logits = accumulated_logits + logits

    assert accumulated_logits is not None
    return accumulated_logits / len(flip_configs)
