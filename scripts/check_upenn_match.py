import os

def check_upenn_folders(base_dir):
    segm_dir = os.path.join(base_dir, 'images_segm')
    struct_dir = os.path.join(base_dir, 'images_structural')
    
    if not os.path.exists(segm_dir) or not os.path.exists(struct_dir):
        print(f"Error: Data directories not found in {base_dir}")
        return

    segm_files = [f for f in os.listdir(segm_dir) if f.endswith('_segm.nii.gz')]
    patient_ids = [f.replace('_segm.nii.gz', '') for f in segm_files]
    
    print(f"Total patients with segmentations: {len(patient_ids)}")
    
    modalities = ['FLAIR', 'T1', 'T1GD', 'T2']
    incomplete_patients = []
    
    for pid in patient_ids:
        pid_struct_dir = os.path.join(struct_dir, pid)
        if not os.path.exists(pid_struct_dir):
            print(f"Missing structural directory for patient {pid}")
            incomplete_patients.append(pid)
            continue
            
        for mod in modalities:
            mod_file = os.path.join(pid_struct_dir, f"{pid}_{mod}.nii.gz")
            if not os.path.exists(mod_file):
                print(f"Missing {mod} for patient {pid}")
                incomplete_patients.append(pid)
                break
                
    print(f"\nSummary for UPENN-GBM Match Check:")
    print(f"Total patients checked: {len(patient_ids)}")
    print(f"Incomplete patients: {len(incomplete_patients)}")
    
    # Check for structural images without segmentations
    all_struct_pids = [d for d in os.listdir(struct_dir) if os.path.isdir(os.path.join(struct_dir, d))]
    pids_without_segm = set(all_struct_pids) - set(patient_ids)
    if pids_without_segm:
        print(f"Patients with structural images but NO segmentations: {len(pids_without_segm)}")
    else:
        print("All patients with structural images have segmentations.")

    if not incomplete_patients:
        print("All segmentations have a complete set of 4 modalities.")

if __name__ == "__main__":
    base = 'data/raw/UPENN-GBM'
    check_upenn_folders(base)
