import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path


def prepare_upenn_dataset():
    raw_dir = Path('data/raw/UPENN-GBM')
    proc_dir = Path('data/processed/UPENN-GBM')

    images_out = proc_dir / 'imagesTr'
    labels_out = proc_dir / 'labelsTr'

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    segm_dir = raw_dir / 'images_segm'
    struct_dir = raw_dir / 'images_structural'

    # Модальности в порядке, соответствующем MSD
    modalities = ['FLAIR', 'T1', 'T1GD', 'T2']

    patient_files = [f for f in os.listdir(segm_dir) if f.endswith('_segm.nii.gz')]

    dataset_info = {
        "name": "UPENN-GBM-4D",
        "description": "UPENN-GBM dataset with 4D stacked modalities and MSD-compatible labels",
        "reference": "University of Pennsylvania Glioblastoma Cholangiocarcinoma Dataset",
        "modality": {"0": "FLAIR", "1": "T1", "2": "T1GD", "3": "T2"},
        "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"},
        "numTraining": len(patient_files),
        "training": []
    }

    print(f"Processing {len(patient_files)} patients...")

    for segm_file in sorted(patient_files):
        pid = segm_file.replace('_segm.nii.gz', '')

        # 1. Stack Modalities
        mod_images = []
        for mod in modalities:
            mod_path = struct_dir / pid / f"{pid}_{mod}.nii.gz"
            img = nib.load(mod_path)
            # Bring to RAS orientation
            img = nib.as_closest_canonical(img)
            mod_images.append(img.get_fdata())
            affine = img.affine
            header = img.header

        # Create 4D array (H, W, D, 4)
        stacked_data = np.stack(mod_images, axis=-1)

        # 2. Process Label (Remap to MSD convention)
        segm_path = segm_dir / segm_file
        segm_img = nib.load(segm_path)
        # Bring to RAS orientation
        segm_img = nib.as_closest_canonical(segm_img)
        segm_data = segm_img.get_fdata()

        new_segm_data = np.zeros_like(segm_data)
        new_segm_data[segm_data == 2] = 1  # Edema
        new_segm_data[segm_data == 1] = 2  # Non-enhancing
        new_segm_data[segm_data == 4] = 3  # Enhancing

        img_out_path = images_out / f"{pid}.nii.gz"
        lbl_out_path = labels_out / f"{pid}.nii.gz"

        # Save image (using header and affine from first/last modality - they are all aligned)
        new_img = nib.Nifti1Image(stacked_data.astype(np.float32), affine, header)
        nib.save(new_img, img_out_path)

        # Save label (using header and affine from reoriented segm_img)
        new_lbl = nib.Nifti1Image(new_segm_data.astype(np.uint8), segm_img.affine, segm_img.header)
        nib.save(new_lbl, lbl_out_path)

        dataset_info["training"].append({
            "image": f"./imagesTr/{pid}.nii.gz",
            "label": f"./labelsTr/{pid}.nii.gz"
        })

        print(f"Done: {pid}")

    with open(proc_dir / 'dataset.json', 'w') as f:
        json.dump(dataset_info, f, indent=4)

    print(f"\nSuccessfully prepared {len(patient_files)} cases in {proc_dir}")


if __name__ == "__main__":
    prepare_upenn_dataset()