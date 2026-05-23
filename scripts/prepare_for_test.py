import shutil
import random
from pathlib import Path

def isolate_test_set(processed_dir, test_dir, train_dir, target_sites, test_ratio=0.2, seed=42):
    random.seed(seed)
    
    processed_path = Path(processed_dir).resolve()
    test_path = Path(test_dir).resolve()
    train_path = Path(train_dir).resolve()
    
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed directory not found at: {processed_path}")

    # 1. DELETE OLD FILES: Wipe the train and test folders entirely, then recreate them
    for target_dir in [test_path, train_path]:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    files = []
    
    # 2. ONLY SEARCH SPECIFIED SITES
    for site in target_sites:
        site_path = processed_path / site
        if not site_path.exists():
            print(f"[WARNING] Skipping {site}: Folder not found.")
            continue
            
        # Find all .npz files inside this specific site's folder
        site_files = list(site_path.rglob('*.npz'))
        files.extend(site_files)
    
    total_files = len(files)
    if total_files == 0:
        print(f"No valid .npz files found in the specified sites: {target_sites}")
        return

    print(f"Found {total_files} total patches across selected sites.")

    # Shuffle patches
    random.shuffle(files)
    
    # Calculate split
    num_test_files = int(total_files * test_ratio)
    test_files = files[:num_test_files]
    train_files = files[num_test_files:]

    print(f"Allocating {len(train_files)} patches to train...")
    print(f"Allocating {len(test_files)} patches to test...")

    def get_site_name(filepath):
        # Extract the site name dynamically
        return filepath.relative_to(processed_path).parts[0]

    # Copy files to their respective folders
    for f in test_files:
        site_name = get_site_name(f)
        new_filename = f"{site_name}_{f.name}"
        shutil.copy(f, test_path / new_filename)

    for f in train_files:
        site_name = get_site_name(f)
        new_filename = f"{site_name}_{f.name}"
        shutil.copy(f, train_path / new_filename)

    print("\n[SUCCESS] Preparation Complete!")
    print(f"Test set:  {test_path}")
    print(f"Train set: {train_path}")

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    TEST_DIR = "data/processed/test"
    TRAIN_DIR = "data/processed/train"
    
    # Define exactly which site folders to pull patches from.
    SITES_TO_USE = ["JasperRidge", "MoffetField"]
    
    isolate_test_set(
        processed_dir=PROCESSED_DIR, 
        test_dir=TEST_DIR, 
        train_dir=TRAIN_DIR,
        target_sites=SITES_TO_USE,
        test_ratio=0.2, 
        seed=42
    )