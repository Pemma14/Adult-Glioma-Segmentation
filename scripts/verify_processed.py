import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path
import sys

def verify_dataset(dataset_name):
    proc_dir = Path('data/processed') / dataset_name
    json_path = proc_dir / 'dataset.json'
    
    if not json_path.exists():
        print(f"Error: dataset.json not found in {proc_dir}")
        return False

    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    
    training_cases = dataset_info.get('training', [])
    print(f"\n--- Verifying {dataset_name} ({len(training_cases)} cases) ---")

    errors = []
    
    for case in training_cases:
        img_path = proc_dir / case['image'].lstrip('./')
        lbl_path = proc_dir / case['label'].lstrip('./')
        pid = os.path.basename(img_path).replace('.nii.gz', '')

        if not img_path.exists():
            errors.append(f"{pid}: Missing image file: {img_path}")
            continue
        if not lbl_path.exists():
            errors.append(f"{pid}: Missing label file: {lbl_path}")
            continue

        try:
            img = nib.load(img_path)
            lbl = nib.load(lbl_path)
            
            # 1. Check Dimensions
            if len(img.shape) != 4 or img.shape[3] != 4:
                errors.append(f"{pid}: Image is not 4D or doesn't have 4 channels (Shape: {img.shape})")
            
            if img.shape[:3] != lbl.shape:
                errors.append(f"{pid}: Spatial dimensions mismatch. Img: {img.shape[:3]}, Lbl: {lbl.shape}")

            # 2. Check Affines
            if not np.allclose(img.affine, lbl.affine):
                errors.append(f"{pid}: Affine matrices mismatch")

            # 3. Check Orientation (RAS)
            orientation = nib.aff2axcodes(img.affine)
            if orientation != ('R', 'A', 'S'):
                errors.append(f"{pid}: Wrong orientation: {orientation}. Expected: ('R', 'A', 'S')")

            # 4. Check Labels
            lbl_data = lbl.get_fdata()
            unique_labels = np.unique(lbl_data)
            if not set(unique_labels).issubset({0, 1, 2, 3}):
                errors.append(f"{pid}: Unexpected labels found: {unique_labels}")

        except Exception as e:
            errors.append(f"{pid}: Critical error: {str(e)}")

    if not errors:
        print(f" {dataset_name} passed all checks!")
        return True
    else:
        print(f" {dataset_name} FAILED with {len(errors)} issues:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more.")
        return False

if __name__ == "__main__":
    datasets = ['UPENN-GBM', 'MSD_BrainTumour']
    all_success = True
    for ds in datasets:
        if not verify_dataset(ds):
            all_success = False
    
    if not all_success:
        sys.exit(1)
