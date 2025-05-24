#!/usr/bin/env python3
"""
Setup script to create symbolic links or move the ACDC dataset to the expected location.

Usage:
    python setup_dataset.py /path/to/your/ACDC_preprocessed
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    # Get current directory (where your app_client.py is)
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Default target location for the dataset
    target_dir = current_dir / "ACDC_preprocessed"
    
    # Check if ACDC_preprocessed already exists in the current directory
    if target_dir.exists():
        print(f"Dataset directory already exists at {target_dir}")
        subdirs = [d for d in target_dir.iterdir() if d.is_dir()]
        print(f"Contains subdirectories: {[d.name for d in subdirs]}")
        return
    
    # If command line argument is provided, use it as the source directory
    if len(sys.argv) > 1:
        source_dir = Path(sys.argv[1])
        if not source_dir.exists():
            print(f"Error: Source directory {source_dir} does not exist.")
            return
        
        # Create symbolic link (more efficient than copying)
        try:
            os.symlink(source_dir, target_dir, target_is_directory=True)
            print(f"Created symbolic link from {source_dir} to {target_dir}")
        except OSError as e:
            print(f"Failed to create symbolic link: {e}")
            print("Attempting to copy files instead...")
            
            # If symlink fails, copy the directory
            try:
                shutil.copytree(source_dir, target_dir)
                print(f"Successfully copied {source_dir} to {target_dir}")
            except Exception as e:
                print(f"Failed to copy directory: {e}")
                return
    else:
        print("Please provide the path to your ACDC_preprocessed directory:")
        print("python setup_dataset.py /path/to/your/ACDC_preprocessed")
        return
    
    # Verify the dataset structure
    required_dirs = ["ACDC_testing_volumes", "ACDC_training_slices", "ACDC_training_volumes"]
    missing_dirs = [d for d in required_dirs if not (target_dir / d).exists()]
    
    if missing_dirs:
        print(f"Warning: The following subdirectories are missing in {target_dir}:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("Please check your dataset structure.")
    else:
        print("Dataset structure looks correct!")
        print("Directory structure:")
        for d in required_dirs:
            subdir = target_dir / d
            files = list(subdir.glob("*.h5"))[:5]  # Show first 5 files
            print(f"  - {d}: {len(files)} H5 files" + (f" (e.g., {[f.name for f in files]})" if files else ""))
    
    print("\nYou can now run the Flower simulation:")
    print("flower-simulation --server-app app_server:app --client-app app_client:app --num-supernodes 2 --app-dir .")

if __name__ == "__main__":
    main()
