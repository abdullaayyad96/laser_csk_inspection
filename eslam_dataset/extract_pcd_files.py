#!/usr/bin/env python3
"""
Script to extract Hole-11xx_3d.pcd files from zip archives
Author: GitHub Copilot
Date: October 31, 2025
"""

import os
import zipfile
import shutil
import glob
import re
from pathlib import Path


def extract_pcd_files(source_directory=".", output_directory="extracted_pcd_files"):
    """
    Extract Hole-11xx_3d.pcd files from zip archives.
    
    Args:
        source_directory (str): Directory containing the zip files
        output_directory (str): Directory to save extracted PCD files
    """
    
    # Convert to Path objects for easier handling
    source_path = Path(source_directory).resolve()
    output_path = Path(output_directory).resolve()
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Find all Hole-*.zip files
    zip_pattern = source_path / "Hole-*.zip"
    zip_files = glob.glob(str(zip_pattern))
    
    if not zip_files:
        print("No Hole-*.zip files found in the directory.")
        return
    
    print(f"Found {len(zip_files)} zip files to process")
    
    extracted_count = 0
    
    for zip_file_path in zip_files:
        zip_file = Path(zip_file_path)
        print(f"\nProcessing: {zip_file.name}")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get list of all files in the zip
                file_list = zip_ref.namelist()
                
                # Filter for files ending with '_3d.pcd'
                pcd_files = [f for f in file_list if f.endswith('_3d.pcd')]
                
                for pcd_file in pcd_files:
                    # Extract just the filename from the path
                    filename = Path(pcd_file).name
                    
                    # Check if it matches the expected pattern (Hole-11xx_3d.pcd)
                    if re.match(r'^Hole-\d+_3d\.pcd$', filename):
                        print(f"  Found target file: {filename}")
                        
                        # Extract the file to output directory
                        with zip_ref.open(pcd_file) as source_file:
                            output_file_path = output_path / filename
                            with open(output_file_path, 'wb') as target_file:
                                shutil.copyfileobj(source_file, target_file)
                        
                        print(f"  Copied to: {output_file_path}")
                        extracted_count += 1
                    else:
                        print(f"  Skipping file (doesn't match pattern): {filename}")
                
                if not pcd_files:
                    print("  No _3d.pcd files found in this archive")
                    
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
    extracted_files = list(output_path.glob("*.pcd"))
    if extracted_files:
        print(f"\nExtracted files:")
        for file in sorted(extracted_files):
            print(f"  - {file.name}")
    else:
        print("\nNo files were extracted.")


def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract Hole-11xx_3d.pcd files from zip archives"
    )
    parser.add_argument(
        "--source", "-s",
        default=".",
        help="Source directory containing zip files (default: current directory)"
    )
    parser.add_argument(
        "--output", "-o",
        default="extracted_pcd_files",
        help="Output directory for extracted PCD files (default: extracted_pcd_files)"
    )
    
    args = parser.parse_args()
    
    print("PCD File Extractor")
    print("=" * 50)
    print(f"Source directory: {Path(args.source).resolve()}")
    
    extract_pcd_files(args.source, args.output)


if __name__ == "__main__":
    main()