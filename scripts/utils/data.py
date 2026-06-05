import yaml
from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from monai.data import DataLoader, CacheDataset

def load_config(config_path, base_config_path="configs/base.yaml"):
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
    
    return config

def get_folds(metadata_path, n_splits=5):
    df = pd.read_csv(metadata_path)
    df = df[df['dataset'] == 'MSD_BrainTumour'].copy()
    # Стратификация по объему (делим на 5 квантилей)
    df['volume_bin'] = pd.qcut(df['total_tumor_voxels'], q=5, labels=False, duplicates='drop')
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds = []
    for train_idx, val_idx in skf.split(df, df['volume_bin']):
        folds.append({
            'train': df.iloc[train_idx],
            'val': df.iloc[val_idx]
        })
    return folds

def get_data_dicts(df_subset):
    data_dicts = []
    for _, row in df_subset.iterrows():
        ds = row['dataset']
        data_dicts.append({
            "image": f"data/processed/{ds}/{row['image_path']}",
            "label": f"data/processed/{ds}/{row['label_path']}",
            "case_id": Path(row['image_path']).name.split('.')[0]
        })
    return data_dicts

def get_loaders(config, train_files, val_files, train_transforms, val_transforms):
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=config.get("cache_rate", 1.0),
        num_workers=config.get("num_workers_cache", 4)
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=config.get("cache_rate", 1.0),
        num_workers=config.get("num_workers_cache", 2)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader
