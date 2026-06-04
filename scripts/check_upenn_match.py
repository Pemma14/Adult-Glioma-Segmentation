import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def check_upenn_folders(base_dir):
    base_path = Path(base_dir)
    segm_dir = base_path / 'images_segm'
    struct_dir = base_path / 'images_structural'
    
    if not segm_dir.exists() or not struct_dir.exists():
        logger.error(f"Data directories not found in {base_dir}")
        return

    segm_files = [f.name for f in segm_dir.glob('*_segm.nii.gz')]
    patient_ids = [f.replace('_segm.nii.gz', '') for f in segm_files]
    
    logger.info(f"Total patients with segmentations: {len(patient_ids)}")
    
    modalities = ['FLAIR', 'T1', 'T1GD', 'T2']
    incomplete_patients = []
    
    for pid in patient_ids:
        pid_struct_dir = struct_dir / pid
        if not pid_struct_dir.exists():
            logger.warning(f"Missing structural directory for patient {pid}")
            incomplete_patients.append(pid)
            continue
            
        for mod in modalities:
            mod_file = pid_struct_dir / f"{pid}_{mod}.nii.gz"
            if not mod_file.exists():
                logger.warning(f"Missing {mod} for patient {pid}")
                incomplete_patients.append(pid)
                break
                
    logger.info(f"Summary for UPENN-GBM Match Check:")
    logger.info(f"Total patients checked: {len(patient_ids)}")
    logger.info(f"Incomplete patients: {len(incomplete_patients)}")
    
    # Check for structural images without segmentations
    all_struct_pids = [d.name for d in struct_dir.iterdir() if d.is_dir()]
    pids_without_segm = set(all_struct_pids) - set(patient_ids)
    if pids_without_segm:
        logger.info(f"Patients with structural images but NO segmentations: {len(pids_without_segm)}")
    else:
        logger.info("All patients with structural images have segmentations.")

    if not incomplete_patients:
        logger.info("All segmentations have a complete set of 4 modalities.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    base = 'data/raw/UPENN-GBM'
    check_upenn_folders(base)
