#!/usr/bin/env python3
"""
Profile Dataset Creator
----------------------
Processes JSON profile files to create a structured dataset with:
- Filtered profiles (dy within ±0.2mm)
- Converted coordinates (meters to mm)
- Manual countersink depths from CSV
- Structured arrays for x, y, z coordinates

Author: Abdulla Ayyad <abdullaayyad96@gmail.com>
Date: November 1, 2025
"""

import os
import json
import csv
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


class ProfileDatasetCreator:
    def __init__(self, json_directory: str = "extracted_profile_json_files", 
                 manual_csv_path: str = "manual.csv",
                 output_directory: str = "profile_dataset"):
        """
        Initialize the profile dataset creator
        
        Args:
            json_directory: Directory containing JSON profile files
            manual_csv_path: Path to manual.csv with countersink depths
            output_directory: Directory to save the processed dataset
        """
        self.json_directory = Path(json_directory)
        self.manual_csv_path = Path(manual_csv_path)
        self.output_directory = Path(output_directory)
        
        # Create output directory
        self.output_directory.mkdir(exist_ok=True)
        
        # Load manual countersink depths
        self.manual_depths = self.load_manual_depths()
        
        print(f"📁 JSON directory: {self.json_directory}")
        print(f"📊 Manual CSV: {self.manual_csv_path}")
        print(f"💾 Output directory: {self.output_directory}")
        print(f"🎯 Loaded manual depths for {len(self.manual_depths)} holes")

    def load_manual_depths(self) -> Dict[str, Dict[str, float]]:
        """
        Load manual countersink depths from CSV file
        
        Returns:
            Dictionary with hole numbers as keys and left/right depths as values
        """
        manual_depths = {}
        
        if not self.manual_csv_path.exists():
            print(f"⚠️  Manual CSV file not found: {self.manual_csv_path}")
            return manual_depths
        
        try:
            with open(self.manual_csv_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Read header row
                values = next(reader)   # Read values row
                
                # Parse headers to extract hole numbers and sides
                for i, header in enumerate(headers):
                    # Extract hole number and side (e.g., "1168l" -> hole=1168, side=left)
                    match = re.match(r'^(\d+)([lr])$', header.strip())
                    if match:
                        hole_num = match.group(1)
                        side = 'left' if match.group(2) == 'l' else 'right'
                        
                        if hole_num not in manual_depths:
                            manual_depths[hole_num] = {}
                        
                        try:
                            depth_value = float(values[i])
                            manual_depths[hole_num][side] = depth_value
                        except (ValueError, IndexError):
                            print(f"⚠️  Could not parse depth for {header}: {values[i] if i < len(values) else 'missing'}")
                
                print(f"✅ Loaded manual depths for holes: {sorted(manual_depths.keys())}")
                
        except Exception as e:
            print(f"❌ Error loading manual CSV: {e}")
        
        return manual_depths

    def extract_hole_number(self, filename: str) -> Optional[str]:
        """
        Extract hole number from filename
        
        Args:
            filename: JSON filename (e.g., "1_Hole-1167_profiles.json")
            
        Returns:
            Hole number as string (e.g., "1167") or None if not found
        """
        # Look for pattern like "Hole-1167" in filename
        match = re.search(r'Hole-(\d+)', filename)
        if match:
            return match.group(1)
        return None

    def convert_to_mm(self, value: float) -> float:
        """Convert from meters to millimeters"""
        return value * 1000.0

    def filter_profiles_by_dy(self, profiles: Dict, dy_threshold: float = 0.2) -> Dict:
        """
        Filter profiles to keep only those within dy threshold
        
        Args:
            profiles: Dictionary of profile data
            dy_threshold: Maximum absolute dy value to keep (in mm)
            
        Returns:
            Filtered profiles dictionary
        """
        filtered_profiles = {}
        
        for profile_id, profile_data in profiles.items():
            dy_meters = profile_data.get('dy', 0)
            dy_mm = abs(dy_meters) # dy already in mm
            
            if dy_mm <= dy_threshold:
                filtered_profiles[profile_id] = profile_data
            else:
                print(f"  🚫 Filtered out {profile_id}: dy={dy_mm:.2f}mm > {dy_threshold}mm")
        
        return filtered_profiles

    def convert_profile_to_arrays(self, profile_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert profile points to x, y, z arrays in millimeters
        
        Args:
            profile_data: Profile data with points list
            
        Returns:
            Tuple of (x_array, y_array, z_array) in mm
        """
        points = profile_data.get('points', [])
        
        if not points:
            return np.array([]), np.array([]), np.array([])
        
        # Extract coordinates and convert to mm
        x_coords = []
        y_coords = []
        z_coords = []
        
        for point in points:
            x_coords.append(self.convert_to_mm(point.get('x', 0)))
            y_coords.append(self.convert_to_mm(point.get('y', 0)))
            z_coords.append(self.convert_to_mm(point.get('z', 0)))
        
        return np.array(x_coords), np.array(y_coords), np.array(z_coords)

    def process_json_file(self, json_file_path: Path) -> List[Dict]:
        """
        Process a single JSON profile file
        
        Args:
            json_file_path: Path to JSON file
            
        Returns:
            List of processed profile data dictionaries
        """
        print(f"\n🔄 Processing: {json_file_path.name}")
        
        # Extract hole number from filename
        hole_number = self.extract_hole_number(json_file_path.name)
        if not hole_number:
            print(f"  ⚠️  Could not extract hole number from filename")
            return []
        
        # Get manual depths for this hole
        left_depth = self.manual_depths.get(hole_number, {}).get('left', np.nan)
        right_depth = self.manual_depths.get(hole_number, {}).get('right', np.nan)
        
        # Skip if no manual depth values are available
        if np.isnan(left_depth) or np.isnan(right_depth):
            print(f"  🚫 Skipping Hole {hole_number}: Missing manual depths (Left={left_depth}, Right={right_depth})")
            return []
        
        print(f"  🎯 Hole {hole_number}: Left={left_depth:.4f}mm, Right={right_depth:.4f}mm")
        
        try:
            # Load JSON data
            with open(json_file_path, 'r') as f:
                profiles_data = json.load(f)
            
            # Filter profiles by dy threshold
            filtered_profiles = self.filter_profiles_by_dy(profiles_data, dy_threshold=0.2)
            print(f"  📊 Kept {len(filtered_profiles)}/{len(profiles_data)} profiles (dy ≤ 0.2mm)")
            
            # Process each filtered profile
            processed_profiles = []
            
            for profile_id, profile_data in filtered_profiles.items():
                # Convert to arrays
                x_array, y_array, z_array = self.convert_profile_to_arrays(profile_data)
                
                if len(x_array) == 0:
                    print(f"    ⚠️  Skipping {profile_id}: no points")
                    continue
                
                # Create processed profile data
                processed_profile = {
                    'hole_name': f"Hole-{hole_number}",
                    'hole_number': hole_number,
                    'profile_id': profile_id,
                    'dy_mm': profile_data.get('dy', 0),
                    'x_array_mm': x_array,
                    'y_array_mm': y_array,
                    'z_array_mm': z_array,
                    'left_csk_depth_mm': left_depth,
                    'right_csk_depth_mm': right_depth,
                    'num_points': len(x_array)
                }
                
                processed_profiles.append(processed_profile)
                print(f"    ✅ {profile_id}: dy={processed_profile['dy_mm']:.2f}mm, points={len(x_array)}")
            
            return processed_profiles
            
        except Exception as e:
            print(f"  ❌ Error processing {json_file_path.name}: {e}")
            return []

    def create_dataset(self) -> List[Dict]:
        """
        Create the complete dataset from all JSON files
        
        Returns:
            List of all processed profile data
        """
        print(f"\n🚀 Creating profile dataset...")
        print(f"📁 Searching for JSON files in: {self.json_directory}")
        
        # Find all JSON files
        json_pattern = self.json_directory / "*_profiles.json"
        json_files = list(self.json_directory.glob("*_profiles.json"))
        
        if not json_files:
            print(f"❌ No JSON profile files found in {self.json_directory}")
            return []
        
        print(f"📄 Found {len(json_files)} JSON files")
        
        # Process all files
        all_profiles = []
        for json_file in sorted(json_files):
            profiles = self.process_json_file(json_file)
            all_profiles.extend(profiles)
        
        print(f"\n📊 Dataset creation complete!")
        print(f"✅ Total profiles: {len(all_profiles)}")
        print(f"🎯 Unique holes: {len(set(p['hole_number'] for p in all_profiles))}")
        
        return all_profiles

    def save_dataset(self, dataset: List[Dict], format: str = 'both') -> None:
        """
        Save the dataset in various formats
        
        Args:
            dataset: List of processed profile data
            format: 'json', 'pickle', or 'both'
        """
        if not dataset:
            print("❌ No data to save")
            return
        
        print(f"\n💾 Saving dataset...")
        
        # Save as JSON (without numpy arrays - convert to lists)
        if format in ['json', 'both']:
            json_data = []
            for profile in dataset:
                json_profile = profile.copy()
                # Convert numpy arrays to lists for JSON serialization
                json_profile['x_array_mm'] = profile['x_array_mm'].tolist()
                json_profile['y_array_mm'] = profile['y_array_mm'].tolist()
                json_profile['z_array_mm'] = profile['z_array_mm'].tolist()
                json_data.append(json_profile)
            
            json_file = self.output_directory / "profile_dataset.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"📄 Saved JSON: {json_file}")
        
        # Save as pickle (preserves numpy arrays)
        if format in ['pickle', 'both']:
            import pickle
            pickle_file = self.output_directory / "profile_dataset.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"🥒 Saved pickle: {pickle_file}")
        
        # Save summary statistics
        self.save_dataset_summary(dataset)

    def save_dataset_summary(self, dataset: List[Dict]) -> None:
        """Save dataset summary and statistics"""
        summary_file = self.output_directory / "dataset_summary.txt"
        
        # Calculate statistics
        holes = set(p['hole_number'] for p in dataset)
        total_points = sum(p['num_points'] for p in dataset)
        dy_values = [p['dy_mm'] for p in dataset]
        
        with open(summary_file, 'w') as f:
            f.write("PROFILE DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset Statistics:\n")
            f.write(f"  Total profiles: {len(dataset)}\n")
            f.write(f"  Unique holes: {len(holes)}\n")
            f.write(f"  Total points: {total_points:,}\n")
            f.write(f"  Average points per profile: {total_points/len(dataset):.1f}\n\n")
            
            f.write(f"DY Distribution (mm):\n")
            f.write(f"  Min: {min(dy_values):.3f}\n")
            f.write(f"  Max: {max(dy_values):.3f}\n")
            f.write(f"  Mean: {np.mean(dy_values):.3f}\n")
            f.write(f"  Std: {np.std(dy_values):.3f}\n\n")
            
            f.write(f"Holes included: {sorted(holes)}\n\n")
            
            # Profile count per hole
            hole_counts = {}
            for profile in dataset:
                hole = profile['hole_number']
                hole_counts[hole] = hole_counts.get(hole, 0) + 1
            
            f.write(f"Profiles per hole:\n")
            for hole in sorted(hole_counts.keys()):
                left_depth = next((p['left_csk_depth_mm'] for p in dataset if p['hole_number'] == hole), 'N/A')
                right_depth = next((p['right_csk_depth_mm'] for p in dataset if p['hole_number'] == hole), 'N/A')
                f.write(f"  Hole-{hole}: {hole_counts[hole]} profiles (L:{left_depth:.3f}, R:{right_depth:.3f})\n")
        
        print(f"📋 Saved summary: {summary_file}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create structured dataset from JSON profile files"
    )
    parser.add_argument(
        '--json-dir', '-j',
        default='extracted_profile_json_files',
        help='Directory containing JSON profile files (default: extracted_profile_json_files)'
    )
    parser.add_argument(
        '--manual-csv', '-m',
        default='manual.csv',
        help='Path to manual.csv with countersink depths (default: manual.csv)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='profile_dataset',
        help='Output directory for dataset (default: profile_dataset)'
    )
    parser.add_argument(
        '--dy-threshold', '-t',
        type=float,
        default=0.2,
        help='Maximum absolute dy value in mm to include (default: 0.2)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'pickle', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    
    args = parser.parse_args()
    
    print("PROFILE DATASET CREATOR")
    print("=" * 50)
    
    try:
        # Create dataset creator
        creator = ProfileDatasetCreator(
            json_directory=args.json_dir,
            manual_csv_path=args.manual_csv,
            output_directory=args.output_dir
        )
        
        # Override dy threshold if specified
        creator.dy_threshold = args.dy_threshold
        print(f"🎯 DY threshold: ±{args.dy_threshold}mm")
        
        # Create dataset
        dataset = creator.create_dataset()
        
        if not dataset:
            print("❌ No data to save")
            return 1
        
        # Save dataset
        creator.save_dataset(dataset, format=args.format)
        
        print(f"\n🎉 Dataset creation complete!")
        print(f"📊 {len(dataset)} profiles from {len(set(p['hole_number'] for p in dataset))} holes")
        print(f"💾 Saved to: {creator.output_directory}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())