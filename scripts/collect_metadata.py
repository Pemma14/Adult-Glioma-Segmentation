import os
import json
import nibabel as nib
import csv
import hashlib
from pathlib import Path

def get_flair_hash(img_path):
    """Рассчитывает MD5-хеш для первого канала (FLAIR) изображения."""
    img = nib.load(img_path)
    # Берем данные первого канала (индекс 0 в 4-м измерении)
    data = img.get_fdata()[..., 0]
    return hashlib.md5(data.tobytes()).hexdigest()

def collect_metadata():
    proc_dir = Path('data/processed')
    records = []
    datasets = ['MSD_BrainTumour', 'UPENN-GBM']
    
    print("Collecting metadata and hashes (this may take a few minutes)...")
    
    for ds in datasets:
        ds_path = proc_dir / ds
        json_file = ds_path / 'dataset.json'
        if not json_file.exists():
            print(f"Warning: dataset.json not found for {ds}")
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        training_cases = data.get('training', [])
        for case in training_cases:
            img_rel_path = case['image'].lstrip('./')
            img_path = ds_path / img_rel_path
            pid = os.path.basename(img_rel_path).replace('.nii.gz', '')
            
            try:
                img = nib.load(img_path)
                s = img.shape
                z = img.header.get_zooms()[:3]
                
                # Рассчитываем хеш FLAIR канала для проверки дубликатов
                flair_hash = get_flair_hash(img_path)
                
                records.append([
                    ds, 
                    pid, 
                    s[0], s[1], s[2], 
                    s[3] if len(s) > 3 else 1, 
                    round(float(z[0]), 2), 
                    round(float(z[1]), 2), 
                    round(float(z[2]), 2),
                    flair_hash
                ])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    output_csv = proc_dir / 'metadata.csv'
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'dataset', 'patient_id', 'dim_x', 'dim_y', 'dim_z', 
            'channels', 'spacing_x', 'spacing_y', 'spacing_z', 'flair_hash'
        ])
        w.writerows(records)
        
    print(f"Metadata with hashes saved to {output_csv}")

if __name__ == "__main__":
    collect_metadata()
