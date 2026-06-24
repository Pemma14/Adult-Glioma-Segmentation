"""Clinical inference pipeline for adult glioma segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.data import Dataset
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

from src.utils.transforms import build_postprocess_transform
from src.utils.tta import tta_sliding_window_inference
from src.glioma.model import load_ensemble_for_inference
from src.glioma.settings import load_model_config

logger = logging.getLogger(__name__)


def _prob_to_logit(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert probabilities back to logits for compatibility with post-processing."""
    probs = probs.clamp(min=eps, max=1.0 - eps)
    return torch.log(probs / (1.0 - probs))


def _build_preprocessing_transform(img_size: list[int]) -> Compose:
    """Build the preprocessing transform for a single input image."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image"], spatial_size=img_size),
        ToTensord(keys=["image"]),
    ])


def preprocess_image(image_path: str | Path, img_size: list[int]) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess a NIfTI image for inference.

    Args:
        image_path: Path to the input NIfTI image.
        img_size: Target spatial size for padding.

    Returns:
        Tuple of (preprocessed image tensor of shape (1, C, D, H, W),
                  resampled affine matrix (4x4) from MONAI metadata).
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = _build_preprocessing_transform(img_size)
    dataset = Dataset(data=[{"image": str(image_path)}], transform=transform)
    item = dataset[0]
    image = item["image"]
    if image.ndim == 4:
        image = image.unsqueeze(0)

    resampled_affine = item["image_meta_dict"]["affine"]
    if isinstance(resampled_affine, torch.Tensor):
        resampled_affine = resampled_affine.cpu().numpy()
    return image, resampled_affine


def _ensemble_inference(
    inputs: torch.Tensor,
    models: list[torch.nn.Module],
    roi_size: tuple[int, ...],
    sw_batch_size: int,
    overlap: float,
    overlap_mode: str,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ensemble inference and return mean logits + inter-model uncertainty."""
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
                tta_mode="none",
                use_amp=use_amp,
                return_uncertainty=False,
            )
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            all_probs.append(torch.sigmoid(logits))

    stacked_probs = torch.stack(all_probs, dim=0)  # (M, B, C, D, H, W)
    mean_probs = stacked_probs.mean(dim=0)
    mean_logits = _prob_to_logit(mean_probs)
    uncertainty = torch.std(stacked_probs, dim=0, unbiased=False).mean(dim=1)  # (B, D, H, W)

    return mean_logits, uncertainty


def predict(
    image_path: str | Path,
    output_dir: str | Path,
    device: torch.device | None = None,
    config: dict[str, Any] | None = None,
    folds: list[int] | None = None,
    save_uncertainty: bool = True,
    save_regions: bool = False,
    save_visualization: bool = False,
    n_slices: int = 3,
) -> dict[str, Any]:
    """Run clinical inference on a single image and save results.

    Args:
        image_path: Path to the input NIfTI image.
        output_dir: Directory to save outputs.
        device: Torch device. If ``None``, uses CUDA if available, else CPU.
        config: Model/inference config. If ``None``, loaded from registry.
        save_uncertainty: Whether to save the uncertainty map.
        save_regions: Whether to save per-region binary masks.

    Returns:
        Dictionary with paths to saved outputs and computed volumes.
    """
    if config is None:
        config = load_model_config()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = config["model"]
    inference_cfg = config["inference"]
    img_size = model_cfg["img_size"]
    roi_size = tuple(inference_cfg["roi_size"])

    logger.info("Loading image: %s", image_path)
    inputs, resampled_affine = preprocess_image(image_path, img_size)
    inputs = inputs.to(device)

    logger.info("Loading ensemble models")
    models = load_ensemble_for_inference(device, config, folds=folds)

    logger.info("Running ensemble inference on %s", device)
    use_amp = device.type == "cuda"
    logits, uncertainty = _ensemble_inference(
        inputs,
        models,
        roi_size=roi_size,
        sw_batch_size=inference_cfg["sw_batch_size"],
        overlap=inference_cfg["overlap"],
        overlap_mode=inference_cfg["overlap_mode"],
        use_amp=use_amp,
    )

    postprocess = build_postprocess_transform(
        inference_cfg["postprocess"], inference_cfg["threshold"]
    )
    if postprocess is not None:
        prediction = postprocess(logits)[0].cpu().numpy().astype(np.uint8)
    else:
        prediction = (torch.sigmoid(logits)[0] > inference_cfg["threshold"]).cpu().numpy().astype(np.uint8)

    case_id = image_path.name.replace(".nii.gz", "").replace(".nii", "")

    # Resample prediction from 1mm RAS back to original image space
    original_img = nib.load(str(image_path))
    pred_in_resampled = nib.Nifti1Image(prediction.astype(np.uint8), resampled_affine)
    pred_in_original = nib.processing.resample_from_to(
        pred_in_resampled, original_img, order=0
    )
    prediction_original = pred_in_original.get_fdata().astype(np.uint8)

    from src.glioma.output import save_prediction, save_uncertainty_map
    result = save_prediction(
        prediction_original, case_id, image_path, output_dir, save_regions
    )

    if save_uncertainty:
        unc_in_resampled = nib.Nifti1Image(
            uncertainty[0].cpu().numpy().astype(np.float32), resampled_affine
        )
        unc_in_original = nib.processing.resample_from_to(
            unc_in_resampled, original_img, order=1
        )
        uncertainty_original = unc_in_original.get_fdata().astype(np.float32)
        uncertainty_path = save_uncertainty_map(
            uncertainty_original, case_id, image_path, output_dir
        )
        result["uncertainty_path"] = uncertainty_path

    from src.glioma.report import compute_volumes
    volumes = compute_volumes(prediction_original, reference_nifti=str(image_path))
    result["volumes_ml"] = volumes
    result["case_id"] = case_id

    if save_visualization:
        from src.glioma.output import regions_to_multiclass_mask
        from src.glioma.visualization import save_prediction_visualization

        # Convert preprocessed tensor to (H, W, D, C) for visualization.
        image_for_vis = inputs[0].cpu().numpy()
        image_for_vis = np.transpose(image_for_vis, (2, 3, 1, 0))
        pred_for_vis = np.transpose(
            regions_to_multiclass_mask(prediction), (1, 2, 0)
        )
        unc_for_vis = (
            np.transpose(uncertainty[0].cpu().numpy(), (1, 2, 0))
            if save_uncertainty else None
        )

        vis_dir = output_dir / "visualizations"
        vis_paths = save_prediction_visualization(
            image=image_for_vis,
            prediction=pred_for_vis,
            uncertainty=unc_for_vis,
            case_id=case_id,
            output_dir=vis_dir,
            n_slices=n_slices,
        )
        result["visualization_paths"] = [str(p) for p in vis_paths]

    logger.info("Inference complete for %s", case_id)
    return result
