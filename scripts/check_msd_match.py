import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def check_msd_folders(base_dir):
    base_path = Path(base_dir)
    images_dir = base_path / 'imagesTr'
    labels_dir = base_path / 'labelsTr'
    
    if not images_dir.exists() or not labels_dir.exists():
        logger.error(f"imagesTr or labelsTr not found in {base_dir}")
        return

    images = set(f.name for f in images_dir.glob('*.nii.gz'))
    labels = set(f.name for f in labels_dir.glob('*.nii.gz'))
    
    only_images = images - labels
    only_labels = labels - images
    
    logger.info(f"Checking MSD folder consistency in {base_dir}...")
    logger.info(f"Images found: {len(images)}")
    logger.info(f"Labels found: {len(labels)}")
    
    if not only_images and not only_labels:
        logger.info(f"MSD folders match perfectly. Found {len(images)} cases.")
    else:
        if only_images:
            logger.warning(f"Files only in imagesTr ({len(only_images)}):")
            for f in sorted(list(only_images))[:10]: logger.warning(f"  {f}")
        if only_labels:
            logger.warning(f"Files only in labelsTr ({len(only_labels)}):")
            for f in sorted(list(only_labels))[:10]: logger.warning(f"  {f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    check_msd_folders('data/raw/MSD_BrainTumour')
