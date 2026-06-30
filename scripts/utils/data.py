import yaml
from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from monai.data import DataLoader, CacheDataset
from scripts.prepare_data import ROOT_DIR
from scripts.utils.config_schema import (
    validate_config,
    get_required_training_keys,
    get_required_inference_keys,
)


def load_config(config_path, base_config_path="configs/base.yaml", required_keys=None):
    base_path = Path(base_config_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Критическая ошибка: Базовый конфиг не найден по пути {base_config_path}")

    with open(base_path, "r") as f:
        config = yaml.safe_load(f)

    if config_path:
        spec_path = Path(config_path)
        if not spec_path.exists():
            raise FileNotFoundError(f"Критическая ошибка: Конфиг модели не найден по пути {config_path}")
        with open(spec_path, "r") as f:
            specific_config = yaml.safe_load(f)
            if specific_config:
                config.update(specific_config)

    # If required_keys is explicitly provided, validate immediately. Otherwise callers should
    # validate after determining the model_name.
    if required_keys is not None:
        validate_config(config, required_keys, context=f"конфиге {config_path or base_config_path}")

    return config

def get_folds(metadata_path, n_splits=5):
    df = pd.read_csv(metadata_path)
    msd_df = df[df['dataset'] == 'MSD_BrainTumour'].copy()
    
    if 'fold' not in msd_df.columns:
        raise ValueError(f"Колонка 'fold' не найдена в {metadata_path}. Запустите scripts/fix_metadata_folds.py.")
    
    folds = []
    # Мы используем n_splits из конфига, но колонка fold жестко зафиксирована на 5 фолдов
    # Если в будущем n_splits изменится, нужно будет перегенерировать колонку fold
    for i in range(n_splits):
        train_df = msd_df[msd_df['fold'] != i]
        val_df = msd_df[msd_df['fold'] == i]
        folds.append({
            'train': train_df,
            'val': val_df
        })
    return folds

def get_data_dicts(df_subset):
    data_dicts = []
    for _, row in df_subset.iterrows():
        ds = row['dataset']
        data_dicts.append({
            "image": f"{ROOT_DIR}/data/processed/{ds}/{row['image_path']}",
            "label": f"{ROOT_DIR}/data/processed/{ds}/{row['label_path']}",
            "case_id": Path(row['image_path']).name.split('.')[0]
        })
    return data_dicts

def get_loaders(config, train_files, val_files, train_transforms, val_transforms):
    # These keys must be present in the config; any missing key raises a clear error.
    cache_rate = config["cache_rate"]
    num_workers_cache = config["num_workers_cache"]
    batch_size = config["batch_size"]
    num_workers_loader = config["num_workers_loader"]

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers_cache,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers_cache,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_loader,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=num_workers_loader,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
