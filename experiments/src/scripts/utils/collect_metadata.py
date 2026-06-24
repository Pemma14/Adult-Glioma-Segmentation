import json
import nibabel as nib
import csv
import hashlib
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_flair_hash_from_img(img):
    """Рассчитывает MD5-хеш для первого канала (FLAIR) изображения из уже загруженного объекта."""
    # Берем данные первого канала (индекс 0 в 4-м измерении)
    data = img.get_fdata()[..., 0]
    return hashlib.md5(data.tobytes()).hexdigest()

def collect_metadata():
    proc_dir = Path('data/processed')
    records = []
    datasets = ['MSD_BrainTumour', 'UPENN-GBM']
    
    headers = [
        'dataset', 'patient_id', 'image_path', 'label_path', 
        'dim_x', 'dim_y', 'dim_z', 'channels', 
        'spacing_x', 'spacing_y', 'spacing_z', 'flair_hash',
        'label_0_voxels', 'label_1_voxels', 'label_2_voxels', 'label_3_voxels',
        'total_tumor_voxels', 'fold'
    ]
    
    logger.info("Collecting metadata, hashes and label statistics (this may take a few minutes)...")
    
    for ds in datasets:
        ds_path = proc_dir / ds
        json_file = ds_path / 'dataset.json'
        if not json_file.exists():
            logger.warning(f"dataset.json not found for {ds}")
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        training_cases = data.get('training', [])
        logger.info(f"Processing dataset: {ds}")
        
        for case in tqdm(training_cases):
            img_rel_path = case['image'].lstrip('./')
            label_rel_path = case['label'].lstrip('./')
            img_path = ds_path / img_rel_path
            label_path = ds_path / label_rel_path
            pid = Path(img_rel_path).name.replace('.nii.gz', '')
            
            try:
                # 1. Обработка изображения (размеры, спейсинг, хеш)
                img = nib.load(img_path)
                s = img.shape
                z = img.header.get_zooms()[:3]
                
                # Рассчитываем хеш FLAIR канала для проверки дубликатов
                flair_hash = get_flair_hash_from_img(img)
                
                # 2. Обработка маски (статистика вокселей по классам)
                label_stats = {0: 0, 1: 0, 2: 0, 3: 0}
                if label_path.exists():
                    lbl = nib.load(label_path)
                    lbl_data = lbl.get_fdata()
                    unique, counts = np.unique(lbl_data, return_counts=True)
                    counts_dict = dict(zip(unique.astype(int), counts))
                    for label in label_stats.keys():
                        label_stats[label] = int(counts_dict.get(label, 0))
                else:
                    logger.warning(f"Label not found for patient {pid}: {label_path}")

                # Общий объем опухоли (сумма меток 1, 2 и 3)
                total_tumor = label_stats[1] + label_stats[2] + label_stats[3]
                
                records.append([
                    ds, pid, img_rel_path, label_rel_path,
                    s[0], s[1], s[2], 
                    s[3] if len(s) > 3 else 1, 
                    round(float(z[0]), 2), round(float(z[1]), 2), round(float(z[2]), 2),
                    flair_hash,
                    label_stats[0], label_stats[1], label_stats[2], label_stats[3],
                    total_tumor,
                    -1 # Значение по умолчанию для fold
                ])
            except Exception as e:
                logger.error(f"Error processing {pid} in {ds}: {e}")
                
    output_csv = proc_dir / 'metadata.csv'
    
    # Конвертируем в DataFrame для удобного присвоения фолдов
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    
    df = pd.DataFrame(records, columns=headers)
    
    msd_mask = df['dataset'] == 'MSD_BrainTumour'
    msd_df = df[msd_mask].copy()
    
    if len(msd_df) > 0:
        logger.info("Assigning folds for MSD_BrainTumour...")
        msd_df['volume_bin'] = pd.qcut(msd_df['total_tumor_voxels'], q=5, labels=False, duplicates='drop')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_idx, (_, val_idx) in enumerate(skf.split(msd_df, msd_df['volume_bin'])):
            msd_df.iloc[val_idx, msd_df.columns.get_loc('fold')] = fold_idx
            
        df.loc[msd_mask, 'fold'] = msd_df['fold'].values
        
    df.to_csv(output_csv, index=False)
    logger.info(f"Updated metadata with folds saved to {output_csv}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    collect_metadata()
