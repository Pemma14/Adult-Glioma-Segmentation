import csv
from pathlib import Path
from collections import defaultdict

def check_duplicates():
    metadata_path = Path('data/processed/metadata.csv')
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Run scripts/collect_metadata.py first.")
        return

    hash_to_patients = defaultdict(list)
    
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row['flair_hash']
            patient_info = f"{row['dataset']}/{row['patient_id']}"
            hash_to_patients[h].append(patient_info)
            
    duplicates = {h: patients for h, patients in hash_to_patients.items() if len(patients) > 1}
    
    print(f"Checked {sum(len(p) for p in hash_to_patients.values())} patients.")
    
    if not duplicates:
        print("No duplicates found based on FLAIR hashes.")
    else:
        print(f"Found {len(duplicates)} cases with duplicate hashes:")
        for h, patients in duplicates.items():
            print(f"  Hash {h}: {', '.join(patients)}")

if __name__ == "__main__":
    check_duplicates()
