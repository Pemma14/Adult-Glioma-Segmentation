import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def check_nifti_file(filepath):
    try:
        img = nib.load(filepath)
        header = img.header
        header.get_data_shape()
        data = img.get_fdata(caching='unchanged')

        if np.isnan(data).any():
            return filepath, False, "Contains NaNs"

        return filepath, True, None
    except Exception as e:
        return filepath, False, str(e)


def validate_integrity(base_dirs):
    """Checks if NIfTI files can be loaded and don't contain NaNs."""
    logger.info("--- Checking NIfTI integrity ---")
    nifti_files = []

    for base in base_dirs:
        base_path = Path(base)
        if not base_path.exists():
            logger.warning(f"{base_path} does not exist.")
            continue
        for f in base_path.rglob('*.nii.gz'):
            nifti_files.append(str(f))

    logger.info(f"Checking {len(nifti_files)} files for loadability...")

    broken = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check_nifti_file, nifti_files))

    for path, success, error in results:
        if not success:
            logger.error(f"Broken: {path} | Error: {error}")
            broken.append(path)

    if not broken:
        logger.info("All files are perfectly loadable!")
        return True
    else:
        logger.error(f"Summary: Found {len(broken)} broken files.")
        return False


def validate_duplicates(metadata_path):
    """Checks for duplicate patients based on FLAIR hashes in metadata.csv."""
    logger.info("--- Checking for duplicate patients (FLAIR hashes) ---")
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        logger.error(f"{metadata_path} not found. Run metadata collection first.")
        return False

    hash_to_patients = defaultdict(list)
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row['flair_hash']
            patient_info = f"{row['dataset']}/{row['patient_id']}"
            hash_to_patients[h].append(patient_info)

    duplicates = {h: patients for h, patients in hash_to_patients.items() if len(patients) > 1}

    logger.info(f"Checked {sum(len(p) for p in hash_to_patients.values())} patients.")

    if not duplicates:
        logger.info("No duplicates found based on FLAIR hashes.")
        return True
    else:
        logger.warning(f"Found {len(duplicates)} cases with duplicate hashes:")
        for h, patients in duplicates.items():
            logger.warning(f"  Hash {h}: {', '.join(patients)}")
        return False


def validate_msd_consistency(base_dir):
    """Checks if imagesTr and labelsTr folders in MSD dataset match."""
    logger.info(f"--- Checking MSD folder consistency in {base_dir} ---")
    base_path = Path(base_dir)
    images_dir = base_path / 'imagesTr'
    labels_dir = base_path / 'labelsTr'

    if not images_dir.exists() or not labels_dir.exists():
        logger.error(f"imagesTr or labelsTr not found in {base_dir}")
        return False

    images = set(f.name for f in images_dir.glob('*.nii.gz'))
    labels = set(f.name for f in labels_dir.glob('*.nii.gz'))

    only_images = images - labels
    only_labels = labels - images

    logger.info(f"Images found: {len(images)}, Labels found: {len(labels)}")

    if not only_images and not only_labels:
        logger.info("MSD folders match perfectly.")
        return True
    else:
        if only_images:
            logger.warning(f"Files only in imagesTr ({len(only_images)}):")
            for f in sorted(list(only_images))[:10]: logger.warning(f"  {f}")
        if only_labels:
            logger.warning(f"Files only in labelsTr ({len(only_labels)}):")
            for f in sorted(list(only_labels))[:10]: logger.warning(f"  {f}")
        return False


def validate_upenn_consistency(base_dir):
    """Checks if UPENN-GBM segmentations have all 4 structural modalities."""
    logger.info(f"--- Checking UPENN-GBM folder consistency in {base_dir} ---")
    base_path = Path(base_dir)
    segm_dir = base_path / 'images_segm'
    struct_dir = base_path / 'images_structural'

    if not segm_dir.exists() or not struct_dir.exists():
        logger.error(f"Data directories not found in {base_dir}")
        return False

    segm_files = [f.name for f in segm_dir.glob('*_segm.nii.gz')]
    patient_ids = [f.replace('_segm.nii.gz', '') for f in segm_files]

    modalities = ['FLAIR', 'T1', 'T1GD', 'T2']
    incomplete_patients = []

    for pid in patient_ids:
        pid_struct_dir = struct_dir / pid
        if not pid_struct_dir.exists():
            incomplete_patients.append(pid)
            continue

        for mod in modalities:
            mod_file = pid_struct_dir / f"{pid}_{mod}.nii.gz"
            if not mod_file.exists():
                incomplete_patients.append(pid)
                break

    # Check for structural images without segmentations
    all_struct_pids = [d.name for d in struct_dir.iterdir() if d.is_dir()]
    pids_without_segm = set(all_struct_pids) - set(patient_ids)

    logger.info(f"Total patients with segmentations: {len(patient_ids)}")
    if pids_without_segm:
        logger.info(f"Patients with structural images but NO segmentations: {len(pids_without_segm)}")

    if not incomplete_patients:
        logger.info("All segmentations have a complete set of 4 modalities.")
        return True
    else:
        logger.error(f"Found {len(incomplete_patients)} patients with incomplete modalities.")
        return False


def validate_processed_dataset(dataset_name):
    """Verifies the integrity and correctness of processed datasets."""
    logger.info(f"--- Verifying processed dataset: {dataset_name} ---")
    proc_dir = Path('data/processed') / dataset_name
    json_path = proc_dir / 'dataset.json'

    if not json_path.exists():
        logger.error(f"dataset.json not found in {proc_dir}")
        return False

    with open(json_path, 'r') as f:
        dataset_info = json.load(f)

    training_cases = dataset_info.get('training', [])
    logger.info(f"Verifying {len(training_cases)} cases...")

    errors = []
    for case in training_cases:
        img_path = proc_dir / case['image'].lstrip('./')
        lbl_path = proc_dir / case['label'].lstrip('./')
        pid = img_path.name.replace('.nii.gz', '')

        if not img_path.exists():
            errors.append(f"{pid}: Missing image file: {img_path}")
            continue
        if not lbl_path.exists():
            errors.append(f"{pid}: Missing label file: {lbl_path}")
            continue

        try:
            img = nib.load(img_path)
            lbl = nib.load(lbl_path)

            if len(img.shape) != 4 or img.shape[3] != 4:
                errors.append(f"{pid}: Image is not 4D or doesn't have 4 channels")
            if img.shape[:3] != lbl.shape:
                errors.append(f"{pid}: Spatial dimensions mismatch. Img: {img.shape[:3]}, Lbl: {lbl.shape}")
            if not np.allclose(img.affine, lbl.affine):
                errors.append(f"{pid}: Affine matrices mismatch")
            orientation = nib.aff2axcodes(img.affine)
            if orientation != ('R', 'A', 'S'):
                errors.append(f"{pid}: Wrong orientation: {orientation}")
            lbl_data = lbl.get_fdata()
            if not set(np.unique(lbl_data)).issubset({0, 1, 2, 3}):
                errors.append(f"{pid}: Unexpected labels found")
        except Exception as e:
            errors.append(f"{pid}: Critical error: {str(e)}")

    if not errors:
        logger.info(f"{dataset_name} passed all checks!")
        return True
    else:
        logger.error(f"{dataset_name} FAILED with {len(errors)} issues. First 10:")
        for err in errors[:10]: logger.error(f"  - {err}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate dataset integrity and consistency.")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--integrity", action="store_true", help="Check NIfTI loadability and NaNs")
    parser.add_argument("--duplicates", action="store_true", help="Check for duplicate FLAIR hashes")
    parser.add_argument("--msd", action="store_true", help="Check MSD raw folder consistency")
    parser.add_argument("--upenn", action="store_true", help="Check UPENN-GBM raw folder consistency")
    parser.add_argument("--processed", action="store_true", help="Verify processed datasets (NIfTI, orientation, labels)")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    success = True

    if args.all or args.integrity:
        if not validate_integrity(['data/raw/MSD_BrainTumour', 'data/raw/UPENN-GBM']):
            success = False

    if args.all or args.msd:
        if not validate_msd_consistency('data/raw/MSD_BrainTumour'):
            success = False

    if args.all or args.upenn:
        if not validate_upenn_consistency('data/raw/UPENN-GBM'):
            success = False

    if args.all or args.duplicates:
        if not validate_duplicates('data/processed/metadata.csv'):
            success = False

    if args.all or args.processed:
        for ds in ['UPENN-GBM', 'MSD_BrainTumour']:
            if not validate_processed_dataset(ds):
                success = False

    if not success:
        logger.error("Some validation checks FAILED.")
        sys.exit(1)
    else:
        logger.info("All selected validation checks PASSED.")


if __name__ == "__main__":
    main()