import json
import nibabel as nib
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def collect_label_stats():
    proc_dir = Path('data/processed')
    metadata_path = proc_dir / 'metadata.csv'
    
    if not metadata_path.exists():
        print("Error: metadata.csv not found. Run scripts/collect_metadata.py first.")
        return

    df = pd.read_csv(metadata_path)
    
    # Инициализируем колонки для статистики меток
    for i in range(4):
        df[f'label_{i}_voxels'] = 0
        
    print("Collecting label statistics...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ds = row['dataset']
        pid = row['patient_id']
        
        # Путь к маске
        mask_path = proc_dir / ds / row['label_path']
        
        if not mask_path.exists():
            print(f"Warning: Mask not found for {pid}")
            continue
            
        try:
            lbl = nib.load(mask_path)
            data = lbl.get_fdata()
            
            unique, counts = np.unique(data, return_counts=True)
            label_counts = dict(zip(unique.astype(int), counts))
            
            for label, count in label_counts.items():
                if label < 4:
                    df.at[idx, f'label_{label}_voxels'] = count
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            
    # Добавляем общий объем опухоли (сумму меток 1, 2, 3)
    df['total_tumor_voxels'] = df['label_1_voxels'] + df['label_2_voxels'] + df['label_3_voxels']
    
    # Сохраняем обновленные метаданные
    df.to_csv(metadata_path, index=False)
    print(f"Updated metadata saved to {metadata_path}")

if __name__ == "__main__":
    collect_label_stats()
