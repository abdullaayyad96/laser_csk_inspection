#!/usr/bin/env python3
"""
Script to extract Hole-11xx_profiles.json files from zip archives with dataset prefixes
Author: GitHub Copilot
Date: November 1, 2025
"""

import os
import zipfile
import shutil
import glob
import re
from pathlib import Path


def extract_dataset_name_from_zip(zip_filename):
    """
    Extract dataset name from zip filename to use as prefix.
    
    Args:
        zip_filename (str): Name of the zip file
        
    Returns:
        str: Dataset prefix (e.g., "DS_1", "DS_2", etc.)
    """
    # Try to extract dataset identifier from filename
    # Common patterns: DS_1_Hole-*.zip, Dataset_1_Hole-*.zip, Hole-*_DS1.zip, etc.
    
    zip_name = Path(zip_filename).stem  # Remove .zip extension
    
    # Pattern 1: DS_X_ or Dataset_X_ at the beginning
    match = re.search(r'^(DS_\d+|Dataset_\d+)', zip_name, re.IGNORECASE)
    if match:
        prefix = match.group(1).upper()
        # Normalize to DS_X format
        if prefix.startswith('DATASET_'):
            prefix = prefix.replace('DATASET_', 'DS_')
        return prefix
    
    # Pattern 2: _DSX or _DatasetX at the end
    match = re.search(r'_(DS\d+|Dataset\d+)$', zip_name, re.IGNORECASE)
    if match:
        prefix = match.group(1).upper()
        if prefix.startswith('DATASET'):
            prefix = prefix.replace('DATASET', 'DS_')
        elif re.match(r'^DS\d+$', prefix):
            # Add underscore if missing (DS1 -> DS_1)
            prefix = re.sub(r'^DS(\d+)$', r'DS_\1', prefix)
        return prefix
    
    # Pattern 3: Look for any number in the filename and create DS_X
    numbers = re.findall(r'\d+', zip_name)
    if numbers:
        # Use the first number found
        return f"DS_{numbers[0]}"
    
    # Fallback: use the whole filename (cleaned)
    clean_name = re.sub(r'[^\w]', '_', zip_name)
    return clean_name.upper()


def extract_profile_json_files(source_directory=".", output_directory="extracted_profile_json_files", dataset_name: str = None):
    """
    Extract Hole-11xx_profiles.json files from zip archives with dataset prefixes.

    Args:
        source_directory (str): Directory containing the zip files
        output_directory (str): Directory to save extracted JSON files
        dataset_name (str, optional): If provided, use this dataset prefix (e.g. 'DS_1')
                                     for every extracted file instead of inferring from
                                     the zip filename.
    """
    
    # Convert to Path objects for easier handling
    source_path = Path(source_directory).resolve()
    output_path = Path(output_directory).resolve()
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Find all Hole-*.zip files (or any zip files if none found)
    zip_pattern = source_path / "Hole-*.zip"
    zip_files = glob.glob(str(zip_pattern))
    
    # If no Hole-*.zip files found, look for any zip files
    if not zip_files:
        zip_pattern = source_path / "*.zip"
        zip_files = glob.glob(str(zip_pattern))
    
    if not zip_files:
        print("No zip files found in the directory.")
        return
    
    print(f"Found {len(zip_files)} zip files to process")
    
    extracted_count = 0
    
    for zip_file_path in zip_files:
        zip_file = Path(zip_file_path)
        print(f"\nProcessing: {zip_file.name}")
        
        # Determine dataset prefix: use user-provided name when available,
        # otherwise infer from the zip filename
        if dataset_name:
            dataset_prefix = dataset_name
        else:
            dataset_prefix = extract_dataset_name_from_zip(zip_file.name)
        print(f"  Dataset prefix: {dataset_prefix}")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get list of all files in the zip
                file_list = zip_ref.namelist()
                
                # Filter for files ending with '_profiles.json'
                json_files = [f for f in file_list if f.endswith('_profiles.json')]
                
                for json_file in json_files:
                    # Extract just the filename from the path
                    filename = Path(json_file).name
                    
                    # Check if it matches the expected pattern (Hole-11xx_profiles.json)
                    if re.match(r'^Hole-\d+_profiles\.json$', filename):
                        print(f"  Found target file: {filename}")
                        
                        # Create new filename with dataset prefix
                        new_filename = f"{dataset_prefix}_{filename}"
                        output_file_path = output_path / new_filename
                        
                        # Check if file already exists
                        if output_file_path.exists():
                            print(f"  Warning: {new_filename} already exists, skipping...")
                            continue
                        
                        # Extract the file to output directory
                        with zip_ref.open(json_file) as source_file:
                            with open(output_file_path, 'wb') as target_file:
                                shutil.copyfileobj(source_file, target_file)
                        
                        print(f"  Copied to: {new_filename}")
                        extracted_count += 1
                    else:
                        print(f"  Skipping file (doesn't match pattern): {filename}")
                
                if not json_files:
                    print("  No _profiles.json files found in this archive")
                    
        except zipfile.BadZipFile:
            print(f"  Error: {zip_file.name} is not a valid zip file")
        except Exception as e:
            print(f"  Error processing {zip_file.name}: {str(e)}")
    
    # Summary
    print(f"\n{'='*50}")
    print("Extraction complete!")
    print(f"Total files extracted: {extracted_count}")
    print(f"Files saved to: {output_path}")
    
    # List extracted files
    extracted_files = list(output_path.glob("*.json"))
    if extracted_files:
        print(f"\nExtracted files:")
        for file in sorted(extracted_files):
            print(f"  - {file.name}")
            
        # Group by dataset prefix
        dataset_groups = {}
        for file in extracted_files:
            prefix = file.name.split('_')[0] + '_' + file.name.split('_')[1]
            if prefix not in dataset_groups:
                dataset_groups[prefix] = []
            dataset_groups[prefix].append(file.name)
        
        print(f"\nFiles grouped by dataset:")
        for dataset, files in sorted(dataset_groups.items()):
            print(f"  {dataset}: {len(files)} files")
            for file in sorted(files):
                print(f"    - {file}")
    else:
        print("\nNo files were extracted.")


def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract Hole-11xx_profiles.json files from zip archives with dataset prefixes"
    )
    parser.add_argument(
        "--source", "-s",
        default=".",
        help="Source directory containing zip files (default: current directory)"
    )
    parser.add_argument(
        "--output", "-o",
        default="extracted_profile_json_files",
        help="Output directory for extracted JSON files (default: extracted_profile_json_files)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show what would be extracted without actually extracting files"
    )

    parser.add_argument(
        "--dataset-name", "-n",
        default=None,
        help="Dataset name prefix to use for extracted files (e.g. DS_1). If provided, overrides auto-detection."
    )
    
    args = parser.parse_args()
    
    print("Profile JSON File Extractor")
    print("=" * 50)
    print(f"Source directory: {Path(args.source).resolve()}")
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be extracted")

    extract_profile_json_files(args.source, args.output, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()