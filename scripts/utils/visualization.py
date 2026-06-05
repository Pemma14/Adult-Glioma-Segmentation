import matplotlib.pyplot as plt
import torch
import numpy as np

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
