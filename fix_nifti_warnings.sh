#!/bin/zsh

# Rename problematic files to avoid warnings
# This script renames the files that are causing warnings in the NIfTI loader

# Check if the files exist
if [ -d "ACDC/database/training/patient057" ]; then
    echo "Found patient057 directory, checking for problematic files..."

    # Check if the problematic directories exist
    if [ -d "ACDC/database/training/patient057/patient057_frame09_gt.nii" ]; then
        echo "Found problematic directory patient057_frame09_gt.nii - removing"
        rm -rf "ACDC/database/training/patient057/patient057_frame09_gt.nii"
    fi

    if [ -d "ACDC/database/training/patient057/patient057_frame01_gt.nii" ]; then
        echo "Found problematic directory patient057_frame01_gt.nii - removing"
        rm -rf "ACDC/database/training/patient057/patient057_frame01_gt.nii"
    fi
    
    echo "Done removing problematic directories"
else
    echo "Patient057 directory not found"
fi

echo "Script completed."
