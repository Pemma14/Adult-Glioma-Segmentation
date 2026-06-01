from pathlib import Path
import nibabel as nib
import numpy as np
from concurrent.futures import ThreadPoolExecutor


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


def main():
    base_dirs = [Path('data/raw/MSD_BrainTumour'), Path('data/raw/UPENN-GBM')]
    nifti_files = []

    for base in base_dirs:
        if not base.exists():
            print(f"Warning: {base} does not exist.")
            continue
        for f in base.rglob('*.nii.gz'):
            nifti_files.append(str(f))

    print(f"Checking {len(nifti_files)} files for loadability...")

    broken = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check_nifti_file, nifti_files))

    for path, success, error in results:
        if not success:
            print(f"Broken: {path} | Error: {error}")
            broken.append(path)

    if not broken:
        print("All files are perfectly loadable by NiBabel!")
    else:
        print(f"\nSummary: Found {len(broken)} broken files.")


if __name__ == "__main__":
    main()