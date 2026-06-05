import shutil
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def import_msd():
    raw_dir = Path('data/raw/MSD_BrainTumour')
    proc_dir = Path('data/processed/MSD_BrainTumour')
    
    if not raw_dir.exists():
        logger.error(f"Raw MSD directory not found at {raw_dir}")
        return

    proc_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy and fix dataset.json
    logger.info(f"Processing dataset.json...")
    with open(raw_dir / 'dataset.json', 'r') as f:
        data = json.load(f)
    
    # Unify labels (tumor -> tumour)
    if "labels" in data:
        new_labels = {}
        for k, v in data["labels"].items():
            new_labels[k] = v.replace("tumor", "tumour")
        data["labels"] = new_labels
    
    with open(proc_dir / 'dataset.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    for folder in ['imagesTr', 'labelsTr']:
        dst_folder = proc_dir / folder
        dst_folder.mkdir(exist_ok=True)
        src_folder = raw_dir / folder
        
        logger.info(f"Importing {folder}...")
        files = [f.name for f in src_folder.glob('*.nii.gz')]
        for f in sorted(files):
            src_path = src_folder / f
            dst_path = dst_folder / f
            
            if dst_path.exists():
                continue
                
            # Use hard link to save space and time
            try:
                dst_path.hardlink_to(src_path)
            except (OSError, AttributeError):
                # Fallback to copy if hard link fails (e.g. different partitions or old python)
                shutil.copy2(src_path, dst_path)
                
    logger.info(f"Successfully imported MSD_BrainTumour to {proc_dir}")
    logger.info(f"Total cases: {len(list((proc_dir / 'imagesTr').glob('*.nii.gz')))}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import_msd()
