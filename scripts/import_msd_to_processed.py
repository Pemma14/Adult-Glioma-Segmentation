import shutil
from pathlib import Path

def import_msd():
    raw_dir = Path('data/raw/MSD_BrainTumour')
    proc_dir = Path('data/processed/MSD_BrainTumour')
    
    if not raw_dir.exists():
        print(f"Error: Raw MSD directory not found at {raw_dir}")
        return

    proc_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy dataset.json
    print(f"Copying dataset.json...")
    shutil.copy2(raw_dir / 'dataset.json', proc_dir / 'dataset.json')
    
    for folder in ['imagesTr', 'labelsTr']:
        dst_folder = proc_dir / folder
        dst_folder.mkdir(exist_ok=True)
        src_folder = raw_dir / folder
        
        print(f"Importing {folder}...")
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
                
    print(f"\nSuccessfully imported MSD_BrainTumour to {proc_dir}")
    print(f"Total cases: {len(list((proc_dir / 'imagesTr').glob('*.nii.gz')))}")

if __name__ == "__main__":
    import_msd()
