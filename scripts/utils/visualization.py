import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _to_numpy_array(x):
    """Convert a torch.Tensor or numpy array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _select_center_slice(volume):
    """Return the index of the center slice along the depth axis (last dim)."""
    return volume.shape[-1] // 2


def log_validation_example(case_data, clearml_logger, epoch, fold):
    image = case_data["image"]
    label = case_data["label"]
    prediction = case_data["prediction"]
    case_id = case_data["case_id"]

    # Выбираем центральный срез по глубине
    d = image.shape[-1]
    slice_idx = d // 2

    # Берем первую модальность изображения и первый канал меток (WT)
    img_slice = image[0, :, :, slice_idx].numpy()
    label_slice = label[0, :, :, slice_idx].numpy()
    pred_slice = prediction[0, :, :, slice_idx].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Отрисовка изображения
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"Image ({case_id})")
    axes[0].axis("off")

    # Отрисовка Ground Truth поверх изображения (красным)
    axes[1].imshow(img_slice, cmap="gray")
    axes[1].imshow(label_slice, cmap="Reds", alpha=0.5 * (label_slice > 0))
    axes[1].set_title("Ground Truth (WT)")
    axes[1].axis("off")

    # Отрисовка предсказания поверх изображения (зеленым)
    axes[2].imshow(img_slice, cmap="gray")
    axes[2].imshow(pred_slice, cmap="Greens", alpha=0.5 * (pred_slice > 0))
    axes[2].set_title("Prediction (WT)")
    axes[2].axis("off")

    plt.tight_layout()
    
    # Логируем в ClearML
    clearml_logger.report_matplotlib_figure(
        title="val_examples",
        series=f"fold_{fold}_{case_id}",
        iteration=epoch,
        figure=fig,
        report_image=True
    )
    plt.close(fig)


def log_inference_example(
    image: np.ndarray | torch.Tensor,
    prediction: np.ndarray,
    target: np.ndarray | None,
    case_id: str,
    clearml_logger,
    title: str = "inference_examples",
    series: str | None = None,
) -> None:
    """Log an inference example as a debug image in ClearML.

    The function picks the central axial slice and shows:
    - the original image,
    - the prediction overlay (green),
    - the ground-truth overlay (red) if available.

    Args:
        image: Input image volume of shape (C, D, H, W).
        prediction: Binarized region predictions of shape (3, D, H, W).
        target: Optional binarized region targets of shape (3, D, H, W).
        case_id: Patient/case identifier used in the title.
        clearml_logger: ClearML logger instance.
        title: Debug sample title in ClearML.
        series: Debug sample series name; defaults to ``case_id``.
    """
    image = _to_numpy_array(image)
    prediction = _to_numpy_array(prediction)
    if target is not None:
        target = _to_numpy_array(target)

    slice_idx = _select_center_slice(image)

    # Use the first image modality and the WT channel (index 0) for visualization.
    img_slice = image[0, :, :, slice_idx]
    pred_slice = prediction[0, :, :, slice_idx]
    target_slice = target[0, :, :, slice_idx] if target is not None else None

    ncols = 3 if target is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 2:
        axes = [axes[0], axes[1]]

    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"Image ({case_id})")
    axes[0].axis("off")

    axes[1].imshow(img_slice, cmap="gray")
    axes[1].imshow(pred_slice, cmap="Greens", alpha=0.5 * (pred_slice > 0))
    axes[1].set_title("Prediction (WT)")
    axes[1].axis("off")

    if target_slice is not None:
        axes[2].imshow(img_slice, cmap="gray")
        axes[2].imshow(target_slice, cmap="Reds", alpha=0.5 * (target_slice > 0))
        axes[2].set_title("Ground Truth (WT)")
        axes[2].axis("off")

    plt.tight_layout()

    clearml_logger.report_matplotlib_figure(
        title=title,
        series=series or case_id,
        iteration=0,
        figure=fig,
        report_image=True,
    )
    plt.close(fig)


def _threshold_colors(values, threshold, higher_is_better=True):
    """Return green/red colors based on whether values pass a threshold."""
    if higher_is_better:
        return ["#2ca02c" if v >= threshold else "#d62728" for v in values]
    return ["#2ca02c" if v <= threshold else "#d62728" for v in values]


def plot_dice_summary(results: pd.DataFrame, title: str = "Dice Summary") -> plt.Figure:
    """Create an informative summary figure for Dice metrics.

    Includes:
    - mean Dice per region with a clinical threshold line,
    - per-case sorted mean Dice to quickly spot failing cases.
    """
    dice_cols = ["dice_wt", "dice_tc", "dice_et"]
    region_labels = ["WT", "TC", "ET"]
    dice_threshold = 0.8

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if all(col in results for col in dice_cols):
        mean_dice = results[dice_cols].mean()
        axes[0].bar(
            region_labels,
            mean_dice.values,
            color=_threshold_colors(mean_dice.values, dice_threshold, higher_is_better=True),
        )
        axes[0].axhline(dice_threshold, color="black", linestyle="--", linewidth=1, label=f"threshold = {dice_threshold}")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Mean Dice")
        axes[0].set_title("Mean Dice by Region")
        axes[0].legend()

    if "mean_dice" in results:
        sorted_results = results.sort_values("mean_dice").reset_index(drop=True)
        n_cases = len(sorted_results)
        show_n = min(n_cases, 30)
        show_df = sorted_results.head(show_n) if show_n < n_cases else sorted_results
        colors = _threshold_colors(show_df["mean_dice"].values, dice_threshold, higher_is_better=True)
        axes[1].bar(range(show_n), show_df["mean_dice"].values, color=colors)
        axes[1].axhline(dice_threshold, color="black", linestyle="--", linewidth=1, label=f"threshold = {dice_threshold}")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Mean Dice")
        axes[1].set_xlabel("Case (sorted)")
        axes[1].set_title(f"Per-case Mean Dice (worst {show_n} of {n_cases})" if show_n < n_cases else "Per-case Mean Dice")
        axes[1].set_xticks([])
        axes[1].legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_hd95_summary(results: pd.DataFrame, title: str = "HD95 Summary") -> plt.Figure:
    """Create an informative summary figure for HD95 metrics.

    Includes:
    - mean HD95 per region with a clinical threshold line,
    - per-case sorted mean HD95 to quickly spot failing cases.
    """
    hd95_cols = ["hd95_wt", "hd95_tc", "hd95_et"]
    region_labels = ["WT", "TC", "ET"]
    hd95_threshold = 5.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if all(col in results for col in hd95_cols):
        mean_hd95 = results[hd95_cols].mean()
        axes[0].bar(
            region_labels,
            mean_hd95.values,
            color=_threshold_colors(mean_hd95.values, hd95_threshold, higher_is_better=False),
        )
        axes[0].axhline(hd95_threshold, color="black", linestyle="--", linewidth=1, label=f"threshold = {hd95_threshold}")
        axes[0].set_ylabel("Mean HD95")
        axes[0].set_title("Mean HD95 by Region")
        axes[0].legend()

    if "mean_hd95" in results:
        sorted_results = results.sort_values("mean_hd95", ascending=False).reset_index(drop=True)
        n_cases = len(sorted_results)
        show_n = min(n_cases, 30)
        show_df = sorted_results.head(show_n) if show_n < n_cases else sorted_results
        colors = _threshold_colors(show_df["mean_hd95"].values, hd95_threshold, higher_is_better=False)
        axes[1].bar(range(show_n), show_df["mean_hd95"].values, color=colors)
        axes[1].axhline(hd95_threshold, color="black", linestyle="--", linewidth=1, label=f"threshold = {hd95_threshold}")
        axes[1].set_ylabel("Mean HD95")
        axes[1].set_xlabel("Case (sorted)")
        axes[1].set_title(f"Per-case Mean HD95 (worst {show_n} of {n_cases})" if show_n < n_cases else "Per-case Mean HD95")
        axes[1].set_xticks([])
        axes[1].legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig



