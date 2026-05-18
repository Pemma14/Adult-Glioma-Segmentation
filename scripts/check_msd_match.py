import os

def check_msd_folders(base_dir):
    images_dir = os.path.join(base_dir, 'imagesTr')
    labels_dir = os.path.join(base_dir, 'labelsTr')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: imagesTr or labelsTr not found in {base_dir}")
        return

    images = set(f for f in os.listdir(images_dir) if f.endswith('.nii.gz'))
    labels = set(f for f in os.listdir(labels_dir) if f.endswith('.nii.gz'))
    
    only_images = images - labels
    only_labels = labels - images
    
    print(f"Checking MSD folder consistency in {base_dir}...")
    print(f"Images found: {len(images)}")
    print(f"Labels found: {len(labels)}")
    
    if not only_images and not only_labels:
        print(f"MSD folders match perfectly. Found {len(images)} cases.")
    else:
        if only_images:
            print(f"Files only in imagesTr ({len(only_images)}):")
            for f in sorted(list(only_images))[:10]: print(f"  {f}")
        if only_labels:
            print(f"Files only in labelsTr ({len(only_labels)}):")
            for f in sorted(list(only_labels))[:10]: print(f"  {f}")

if __name__ == "__main__":
    check_msd_folders('data/raw/MSD_BrainTumour')
