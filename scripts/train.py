from pathlib import Path
import sys
import argparse
import logging
import time
import pandas as pd

import torch
import torch.nn as nn
from monai.transforms import AsDiscrete, Activations, Compose
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch
from monai.metrics import DiceMetric

from clearml import Task, Logger
from models import get_model
from scripts.prepare_data import ROOT_DIR

from scripts.utils.data import load_config, get_folds, get_data_dicts, get_loaders
from scripts.utils.transforms import get_transforms
from scripts.utils.visualization import log_validation_example
from scripts.utils.model import load_pretrained_weights, save_checkpoint, load_checkpoint, DeepSupervisionLoss

logger = logging.getLogger(__name__)

# Добавляем корень проекта в путь для импорта моделей
sys.path.append(str(ROOT_DIR))


def train_epoch(model, loader, optimizer, loss_function, device, scaler=None):
    model.train()
    use_amp = scaler is not None
    epoch_loss = torch.tensor(0.0).to(device)
    for batch_data in loader:
        inputs = batch_data["image"].to(device, non_blocking=True)
        
        # Подготовка меток для Deep Supervision
        if isinstance(loss_function, DeepSupervisionLoss) and "label_level_1" in batch_data:
            labels = [batch_data["label"].to(device, non_blocking=True)]
            for i in range(1, 5):
                key = f"label_level_{i}"
                if key in batch_data:
                    labels.append(batch_data[key].to(device, non_blocking=True))
        else:
            labels = batch_data["label"].to(device, non_blocking=True)
            
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        epoch_loss += loss.detach()
    return epoch_loss.item() / len(loader)

def validate(model, loader, device, dice_metric, config, loss_function=None):
    model.eval()
    val_loss = 0.0
    first_case = None
    use_amp = torch.cuda.is_available()
    with torch.no_grad():
        for i, val_data in enumerate(loader):
            val_inputs, val_labels = (
                val_data["image"].to(device, non_blocking=True),
                val_data["label"].to(device, non_blocking=True),
            )
            roi_size = config["img_size"]
            sw_batch_size = config["sw_batch_size"]
            with torch.amp.autocast("cuda", enabled=use_amp):
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            
            if loss_function is not None:
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()

            if i == 0:
                first_case = {
                    "image": val_inputs[0].cpu(),
                    "label": val_labels[0].cpu(),
                    "prediction": (torch.sigmoid(val_outputs[0]) > 0.5).cpu(),
                    "case_id": val_data.get("case_id", ["first_case"])[0]
                }

            # Применяем сигмоиду и порог для метрики
            post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            
            dice_metric(y_pred=val_outputs, y=val_labels)
        
        val_dice = dice_metric.aggregate().item()
        val_dice_per_class = dice_metric.aggregate(reduction="mean_batch")
        dice_metric.reset()
    
    if loss_function is not None:
        return val_dice, val_loss / len(loader), val_dice_per_class, first_case
    return val_dice, None, val_dice_per_class, first_case


def train(config, train_files, val_files, fold=0, resume_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clearml_logger = Logger.current_logger()
    
    # 1. Transforms
    train_transforms, val_transforms = get_transforms(config)

    # 2. Model
    model = get_model(config["model_name"], config).to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config['model_name']}, Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # 3. Transfer Learning
    if config["transfer_learning"] and config["model_name"] in ["swin_unetr", "swin_der"]:
        model = load_pretrained_weights(model, config["model_name"], config["pretrained_path"])
    
    # 4. Loss & Optimizer
    dice_ce_loss = DiceCELoss(to_onehot_y=False, sigmoid=True) # Region-based targets are multi-label
    if config["model_name"] in ["swin_der", "swin_unetr"] and config["deep_supervision"]:
        loss_function = DeepSupervisionLoss(dice_ce_loss)
    else:
        loss_function = dice_ce_loss
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    # 5. Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"])
    
    # 6. Data
    train_loader, val_loader = get_loaders(config, train_files, val_files, train_transforms, val_transforms)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    # AMP (Automatic Mixed Precision)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0
    best_epoch = -1
    if resume_checkpoint:
        start_epoch, best_dice = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, scaler
        )
        best_epoch = start_epoch - 1
        logger.info(f"Resuming training from epoch {start_epoch}, best_dice={best_dice:.4f}")
    
    for epoch in range(start_epoch, config["max_epochs"]):
        epoch_start = time.monotonic()
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device, scaler=scaler)
        epoch_time = time.monotonic() - epoch_start

        clearml_logger.report_scalar("Loss", "train", iteration=epoch, value=train_loss)
        clearml_logger.report_scalar("Learning Rate", "lr", iteration=epoch, value=optimizer.param_groups[0]["lr"])
        clearml_logger.report_scalar("Time", "epoch_sec", iteration=epoch, value=epoch_time)
        
        logger.info(f"Fold {fold}, Epoch {epoch} completed. Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s")

        if (epoch + 1) % config["val_interval"] == 0:
            val_dice, val_loss, val_dice_per_class, first_case = validate(
                model, val_loader, device, dice_metric, config, loss_function=dice_ce_loss
            )
            
            clearml_logger.report_scalar("Val Dice", "mean_dice", iteration=epoch, value=val_dice)
            clearml_logger.report_scalar("Loss", "val", iteration=epoch, value=val_loss)
            
            # Логируем Dice по классам (WT, TC, ET)
            class_names = ["WT", "TC", "ET"]
            for i, class_name in enumerate(class_names):
                if i < len(val_dice_per_class):
                    clearml_logger.report_scalar("Per-class Dice", class_name, iteration=epoch, value=val_dice_per_class[i].item())

            clearml_logger.report_scalar("Best Val Dice so far", "best_dice", iteration=epoch, value=max(best_dice, val_dice))
            
            per_class_info = ", ".join([f"{name}: {val.item():.4f}" for name, val in zip(class_names, val_dice_per_class)])
            logger.info(f"Fold {fold}, Epoch {epoch} Validation Dice: {val_dice:.4f}, Loss: {val_loss:.4f} ({per_class_info})")
            
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                save_checkpoint(
                    model, config, fold,
                    optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, epoch=epoch, best_dice=best_dice
                )
                if first_case is not None:
                    log_validation_example(first_case, clearml_logger, epoch, fold)

            # Early stopping
            if config.get("patience"):
                epochs_without_improvement = epoch - best_epoch
                if epochs_without_improvement > config["patience"] * config["val_interval"]:
                    logger.info(f"Early stopping at epoch {epoch}. Best dice: {best_dice:.4f} at epoch {best_epoch}")
                    break
        
        scheduler.step()

    # Итоговые значения для удобства сравнения в таблице
    clearml_logger.report_single_value("best_val_dice", best_dice)
    clearml_logger.report_single_value("best_epoch", best_epoch)

    return best_dice, best_epoch

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Train Adult Glioma Segmentation Model")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Path to the base config file")
    parser.add_argument("--config", type=str, default="configs/unet3d.yaml", help="Path to the specific config file")
    parser.add_argument("--fold", type=str, default="0", help="Fold index to train (0-4), list of indices (0,1,2), or 'all'")
    parser.add_argument("--stage", type=str, default="research", choices=["research", "hpo", "final", "cv"], help="Experiment stage")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for the task name")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, help="Override weight decay")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--img_size", type=int, nargs=3, help="Override image size (e.g., --img_size 128 128 128)")
    parser.add_argument("--max_epochs", type=int, help="Override max epochs")
    parser.add_argument("--patience", type=int, default="10", help="Early stopping patience (in val_intervals)")
    parser.add_argument("--val_interval", type=int, help="Override validation interval (epochs)")
    parser.add_argument("--sw_batch_size", type=int, help="Override sliding window batch size for validation")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--comment", type=str, default="", help="Experiment comment")
    parser.add_argument("--status", type=str, default="", help="Experiment status (e.g., baseline, trash, candidate, best_so_far, final)")
    args = parser.parse_args()

    # Загружаем конфигурацию
    config = load_config(args.config, args.base_config)
    if args.lr:
        config["lr"] = args.lr
    if args.weight_decay:
        config["weight_decay"] = args.weight_decay
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.img_size:
        config["img_size"] = args.img_size
    if args.max_epochs:
        config["max_epochs"] = args.max_epochs
    if args.patience:
        config["patience"] = args.patience
    if args.val_interval:
        config["val_interval"] = args.val_interval
    if args.sw_batch_size:
        config["sw_batch_size"] = args.sw_batch_size

    # Настройка ClearML
    task_name = f"{args.stage}_{config['model_name']}_f{args.fold}{args.suffix}"
    task = Task.init(
        project_name='AdultGliomaSegmentation', 
        task_name=task_name,
        task_type=Task.TaskTypes.training
    )
    task.connect(config)
    
    # Установка тегов
    tags = [
        f"model:{config['model_name']}",
        f"fold:{args.fold}",
        f"stage:{args.stage}",
        f"status:{args.status}",
    ]

    task.set_tags(tags)

    if args.comment:
        task.set_comment(args.comment)

    # Получаем фолды
    metadata_path = ROOT_DIR / "data/processed/metadata.csv"
    if not Path(metadata_path).exists():
        logger.error(f"Metadata not found at {metadata_path}. Run collection scripts first.")
        sys.exit(1)
        
    folds_data = get_folds(metadata_path, n_splits=config["n_splits"])
    
    if args.fold.lower() == "all":
        fold_indices = list(range(len(folds_data)))
    else:
        try:
            fold_indices = [int(f.strip()) for f in args.fold.split(",")]
        except ValueError:
            logger.error(f"Invalid fold format: {args.fold}. Use integer, comma-separated integers, or 'all'.")
            sys.exit(1)

    for f_idx in fold_indices:
        if f_idx >= len(folds_data):
            logger.error(f"Fold index {f_idx} out of range (0-{len(folds_data)-1})")
            sys.exit(1)

    summary_results = []

    for f_idx in fold_indices:
        logger.info(f"Starting training for fold {f_idx}")
        current_fold = folds_data[f_idx]
        train_files = get_data_dicts(current_fold['train'])
        val_files = get_data_dicts(current_fold['val'])
        
        logger.info(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")

        best_dice, best_epoch = train(config, train_files, val_files, fold=f_idx, resume_checkpoint=args.resume_checkpoint)
        
        summary_results.append({
            "fold": f_idx,
            "best_val_dice": round(best_dice, 4),
            "best_epoch": best_epoch,
            "train_samples": len(train_files),
            "val_samples": len(val_files)
        })

    # Вывод сводной таблицы
    df_summary = pd.DataFrame(summary_results)
    print("\n" + "="*50)
    print("SUMMARY TABLE BY FOLDS")
    print("="*50)
    print(df_summary.to_string(index=False))
    print("="*50 + "\n")

    # Логируем таблицу в ClearML
    task.get_logger().report_table(
        "Cross-Validation Summary", 
        "summary_table", 
        iteration=0, 
        table_plot=df_summary
    )

if __name__ == "__main__":
    main()
