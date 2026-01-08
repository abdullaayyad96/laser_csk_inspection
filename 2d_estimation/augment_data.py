#!/usr/bin/env python3
"""
Data Augmentation Script
-----------------------
Performs data augmentation on the training dataset to increase its size.
Applies four types of augmentations:
1. Z-offset: Adding constant values (±5mm) to all z_array values
2. Z-noise: Adding random Gaussian noise (0.001mm std dev) to z_array values
3. X-shift: Shifting x values by constant amounts (±0.1mm)
4. Y-rotation: Rotating around the y-axis (±0.5 degrees)

The augmentation can multiply the training dataset size by up to 5x by creating 
augmented versions of each original profile based on enabled augmentation types.

Author: Abdulla Ayyad <abdullaayyad96@gmail.com>
Date: November 1, 2025
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random
import copy


class DataAugmentator:
    def __init__(self, split_dataset_directory: str = "split_dataset", 
                 output_directory: str = "augmented_dataset",
                 random_seed: int = 42,
                 enable_z_offset: bool = True,
                 enable_x_shift: bool = True, 
                 enable_y_rotation: bool = True,
                 enable_z_noise: bool = True):
        """
        Initialize the data augmentator
        
        Args:
            split_dataset_directory: Directory containing split datasets
            output_directory: Directory to save augmented datasets
            random_seed: Random seed for reproducible augmentation
            enable_z_offset: Whether to apply z-offset augmentation
            enable_x_shift: Whether to apply x-shift augmentation
            enable_y_rotation: Whether to apply y-rotation augmentation
            enable_z_noise: Whether to apply z-noise augmentation
        """
        self.split_dataset_directory = Path(split_dataset_directory)
        self.output_directory = Path(output_directory)
        self.random_seed = random_seed
        self.enable_z_offset = enable_z_offset
        self.enable_x_shift = enable_x_shift
        self.enable_y_rotation = enable_y_rotation
        self.enable_z_noise = enable_z_noise
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directory
        self.output_directory.mkdir(exist_ok=True)
        
        print(f"📁 Split dataset directory: {self.split_dataset_directory}")
        print(f"💾 Output directory: {self.output_directory}")
        print(f"🎲 Random seed: {random_seed}")
        
        # Load training dataset
        self.train_dataset = self.load_training_dataset()
        print(f"📊 Loaded {len(self.train_dataset)} training profiles")

    def load_training_dataset(self) -> List[Dict]:
        """Load the training dataset from pickle or JSON file"""
        # Try to load from pickle first (preserves numpy arrays)
        pickle_path = self.split_dataset_directory / "train_dataset.pkl"
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    dataset = pickle.load(f)
                print(f"✅ Loaded training dataset from pickle: {pickle_path.name}")
                return dataset
            except Exception as e:
                print(f"⚠️  Failed to load pickle file: {e}")
        
        # Fall back to JSON
        json_path = self.split_dataset_directory / "train_dataset.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    dataset = json.load(f)
                
                # Convert lists back to numpy arrays
                for profile in dataset:
                    profile['x_array_mm'] = np.array(profile['x_array_mm'])
                    profile['y_array_mm'] = np.array(profile['y_array_mm'])
                    profile['z_array_mm'] = np.array(profile['z_array_mm'])
                
                print(f"✅ Loaded training dataset from JSON: {json_path.name}")
                return dataset
            except Exception as e:
                print(f"❌ Failed to load JSON file: {e}")
        
        raise FileNotFoundError(f"No valid training dataset files found in {self.split_dataset_directory}")

    def apply_z_offset(self, profile: Dict, z_offset_mm: float) -> Dict:
        """
        Apply z-offset augmentation: add constant value to all z coordinates
        
        Args:
            profile: Original profile data
            z_offset_mm: Offset to add to all z values (in mm)
            
        Returns:
            Augmented profile with z-offset applied
        """
        augmented_profile = copy.deepcopy(profile)
        
        # Apply z-offset to all z coordinates
        augmented_profile['z_array_mm'] = profile['z_array_mm'] + z_offset_mm
        
        # Update profile metadata
        augmented_profile['augmentation_type'] = 'z_offset'
        augmented_profile['augmentation_params'] = {'z_offset_mm': z_offset_mm}
        augmented_profile['profile_id'] = f"{profile['profile_id']}_z_offset_{z_offset_mm:+.1f}mm"
        
        return augmented_profile

    def apply_x_shift(self, profile: Dict, x_shift_mm: float) -> Dict:
        """
        Apply x-shift augmentation: shift all x coordinates by constant value
        
        Args:
            profile: Original profile data
            x_shift_mm: Shift to add to all x values (in mm)
            
        Returns:
            Augmented profile with x-shift applied
        """
        augmented_profile = copy.deepcopy(profile)
        
        # Apply x-shift to all x coordinates
        augmented_profile['x_array_mm'] = profile['x_array_mm'] + x_shift_mm
        
        # Update profile metadata
        augmented_profile['augmentation_type'] = 'x_shift'
        augmented_profile['augmentation_params'] = {'x_shift_mm': x_shift_mm}
        augmented_profile['profile_id'] = f"{profile['profile_id']}_x_shift_{x_shift_mm:+.1f}mm"
        
        return augmented_profile

    def apply_y_rotation(self, profile: Dict, rotation_degrees: float) -> Dict:
        """
        Apply y-rotation augmentation: rotate around y-axis
        
        Args:
            profile: Original profile data
            rotation_degrees: Rotation angle in degrees
            
        Returns:
            Augmented profile with y-rotation applied
        """
        augmented_profile = copy.deepcopy(profile)
        
        # Convert angle to radians
        rotation_rad = np.radians(rotation_degrees)
        
        # Get original coordinates
        x_coords = profile['x_array_mm']
        z_coords = profile['z_array_mm']
        y_coords = profile['y_array_mm']  # y coordinates remain unchanged
        
        # Apply rotation around y-axis (rotation matrix)
        # x' = x*cos(θ) + z*sin(θ)
        # z' = -x*sin(θ) + z*cos(θ)
        cos_theta = np.cos(rotation_rad)
        sin_theta = np.sin(rotation_rad)
        
        x_rotated = x_coords * cos_theta + z_coords * sin_theta
        z_rotated = -x_coords * sin_theta + z_coords * cos_theta
        
        # Update coordinates
        augmented_profile['x_array_mm'] = x_rotated
        augmented_profile['z_array_mm'] = z_rotated
        # y_array_mm remains unchanged
        
        # Update profile metadata
        augmented_profile['augmentation_type'] = 'y_rotation'
        augmented_profile['augmentation_params'] = {'rotation_degrees': rotation_degrees}
        augmented_profile['profile_id'] = f"{profile['profile_id']}_y_rot_{rotation_degrees:+.1f}deg"
        
        return augmented_profile

    def apply_z_noise(self, profile: Dict, noise_power_mm: float) -> Dict:
        """
        Apply z-noise augmentation: add random noise to all z coordinates
        
        Args:
            profile: Original profile data
            noise_power_mm: Standard deviation of Gaussian noise to add (in mm)
            
        Returns:
            Augmented profile with z-noise applied
        """
        augmented_profile = copy.deepcopy(profile)
        
        # Generate random noise for each z coordinate
        num_points = len(profile['z_array_mm'])
        noise = np.random.normal(0, noise_power_mm, num_points)
        
        # Apply noise to z coordinates
        augmented_profile['z_array_mm'] = profile['z_array_mm'] + noise
        
        # Update profile metadata
        augmented_profile['augmentation_type'] = 'z_noise'
        augmented_profile['augmentation_params'] = {'noise_power_mm': noise_power_mm}
        augmented_profile['profile_id'] = f"{profile['profile_id']}_z_noise_{noise_power_mm:.3f}mm"
        
        return augmented_profile

    def generate_augmentation_parameters(self) -> List[Tuple[str, Dict]]:
        """
        Generate augmentation parameters based on enabled augmentation types
        
        Returns:
            List of (augmentation_type, parameters) tuples
        """
        augmentation_configs = []
        
        # Add z-offset augmentation if enabled
        if self.enable_z_offset:
            augmentation_configs.append(
                ('z_offset', {'z_offset_range': (-5.0, 5.0)})
            )
        
        # Add z-noise augmentation if enabled
        if self.enable_z_noise:
            augmentation_configs.append(
                ('z_noise', {'noise_power_mm': 0.001})
            )
        
        # Add combined x-shift and y-rotation if both are enabled
        if self.enable_x_shift and self.enable_y_rotation:
            augmentation_configs.append(
                ('x_shift_y_rotation', {
                    'x_shift_range': (-0.1, 0.1),
                    'rotation_range': (-0.5, 0.5)
                })
            )
        # Add only x-shift if enabled but y-rotation is disabled
        elif self.enable_x_shift:
            augmentation_configs.append(
                ('x_shift', {'x_shift_range': (-0.1, 0.1)})
            )
        # Add only y-rotation if enabled but x-shift is disabled
        elif self.enable_y_rotation:
            augmentation_configs.append(
                ('y_rotation', {'rotation_range': (-0.5, 0.5)})
            )
        
        return augmentation_configs

    def augment_profile(self, profile: Dict, augmentation_type: str, params: Dict) -> Dict:
        """
        Apply augmentation to a single profile
        
        Args:
            profile: Original profile data
            augmentation_type: Type of augmentation to apply
            params: Augmentation parameters
            
        Returns:
            Augmented profile
        """
        if augmentation_type == 'z_offset':
            # Generate random z-offset within range
            z_min, z_max = params['z_offset_range']
            z_offset = random.uniform(z_min, z_max)
            return self.apply_z_offset(profile, z_offset)
        
        elif augmentation_type == 'z_noise':
            # Apply z-noise with specified power
            noise_power = params['noise_power_mm']
            return self.apply_z_noise(profile, noise_power)
        
        elif augmentation_type == 'x_shift_y_rotation':
            # Apply both x-shift and y-rotation
            x_min, x_max = params['x_shift_range']
            rot_min, rot_max = params['rotation_range']
            
            x_shift = random.uniform(x_min, x_max)
            rotation = random.uniform(rot_min, rot_max)
            
            # Apply x-shift first
            augmented = self.apply_x_shift(profile, x_shift)
            # Then apply y-rotation
            augmented = self.apply_y_rotation(augmented, rotation)
            
            # Update metadata for combined augmentation
            augmented['augmentation_type'] = 'x_shift_y_rotation'
            augmented['augmentation_params'] = {
                'x_shift_mm': x_shift,
                'rotation_degrees': rotation
            }
            augmented['profile_id'] = f"{profile['profile_id']}_x{x_shift:+.1f}_rot{rotation:+.1f}"
            
            return augmented
        
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")

    def augment_training_dataset(self) -> List[Dict]:
        """
        Augment the entire training dataset to triple its size
        
        Returns:
            Augmented training dataset (original + 2 augmented versions per profile)
        """
        print(f"\n🚀 Starting data augmentation...")
        print(f"📊 Original training dataset: {len(self.train_dataset)} profiles")
        
        # Generate augmentation configurations
        aug_configs = self.generate_augmentation_parameters()
        print(f"🔧 Augmentation configurations: {len(aug_configs)}")
        for i, (aug_type, params) in enumerate(aug_configs, 1):
            print(f"  {i}. {aug_type}: {params}")
        
        # Create augmented dataset starting with original profiles
        augmented_dataset = copy.deepcopy(self.train_dataset)
        
        # Add original marker to existing profiles
        for profile in augmented_dataset:
            profile['augmentation_type'] = 'original'
            profile['augmentation_params'] = {}
        
        # Apply each augmentation configuration to all original profiles
        for aug_type, params in aug_configs:
            print(f"\n🔄 Applying {aug_type} augmentation...")
            
            augmented_profiles = []
            for i, profile in enumerate(self.train_dataset):
                try:
                    augmented_profile = self.augment_profile(profile, aug_type, params)
                    augmented_profiles.append(augmented_profile)
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1}/{len(self.train_dataset)} profiles...")
                        
                except Exception as e:
                    print(f"  ⚠️  Failed to augment profile {profile['profile_id']}: {e}")
            
            augmented_dataset.extend(augmented_profiles)
            print(f"  ✅ Added {len(augmented_profiles)} augmented profiles")
        
        print(f"\n📈 Augmentation complete!")
        print(f"  Original profiles: {len(self.train_dataset)}")
        print(f"  Augmented profiles: {len(augmented_dataset) - len(self.train_dataset)}")
        print(f"  Total profiles: {len(augmented_dataset)}")
        print(f"  Multiplication factor: {len(augmented_dataset) / len(self.train_dataset):.1f}x")
        
        return augmented_dataset

    def save_augmented_dataset(self, augmented_dataset: List[Dict]) -> None:
        """Save the augmented training dataset"""
        
        print(f"\n💾 Saving augmented training dataset ({len(augmented_dataset)} profiles)...")
        
        # Save as pickle (preserves numpy arrays)
        pickle_file = self.output_directory / "augmented_train_dataset.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(augmented_dataset, f)
        print(f"  🥒 Saved pickle: {pickle_file.name}")
        
        # Save as JSON (convert numpy arrays to lists)
        json_data = []
        for profile in augmented_dataset:
            json_profile = profile.copy()
            json_profile['x_array_mm'] = profile['x_array_mm'].tolist()
            json_profile['y_array_mm'] = profile['y_array_mm'].tolist()
            json_profile['z_array_mm'] = profile['z_array_mm'].tolist()
            json_data.append(json_profile)
        
        json_file = self.output_directory / "augmented_train_dataset.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"  📄 Saved JSON: {json_file.name}")

    def save_augmentation_summary(self, augmented_dataset: List[Dict]) -> None:
        """Save summary of the data augmentation"""
        
        summary_file = self.output_directory / "augmentation_summary.txt"
        
        # Count augmentation types
        aug_type_counts = {}
        for profile in augmented_dataset:
            aug_type = profile.get('augmentation_type', 'unknown')
            aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1
        
        with open(summary_file, 'w') as f:
            f.write("DATA AUGMENTATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Original training profiles: {len(self.train_dataset)}\n")
            f.write(f"Augmented training profiles: {len(augmented_dataset)}\n")
            f.write(f"Multiplication factor: {len(augmented_dataset) / len(self.train_dataset):.1f}x\n\n")
            
            # Augmentation type breakdown
            f.write("AUGMENTATION TYPE BREAKDOWN\n")
            f.write("-" * 30 + "\n")
            for aug_type, count in sorted(aug_type_counts.items()):
                percentage = count / len(augmented_dataset) * 100
                f.write(f"{aug_type:20s}: {count:5d} profiles ({percentage:5.1f}%)\n")
            
            f.write(f"\nAUGMENTATION PARAMETERS\n")
            f.write("-" * 25 + "\n")
            f.write("Z-offset range: ±5.0 mm\n")
            f.write("Z-noise power: 0.001 mm (std dev)\n")
            f.write("X-shift range: ±0.1 mm\n")
            f.write("Y-rotation range: ±0.5 degrees\n\n")
            
            # Sample augmented profiles
            f.write("SAMPLE AUGMENTED PROFILES\n")
            f.write("-" * 25 + "\n")
            sample_count = 0
            for profile in augmented_dataset:
                if profile.get('augmentation_type') != 'original' and sample_count < 10:
                    f.write(f"Profile: {profile['profile_id']}\n")
                    f.write(f"  Type: {profile['augmentation_type']}\n")
                    f.write(f"  Params: {profile['augmentation_params']}\n")
                    f.write(f"  Points: {profile['num_points']}\n\n")
                    sample_count += 1
        
        print(f"📋 Saved augmentation summary: {summary_file.name}")

    def augment_and_save(self) -> None:
        """Main method to perform augmentation and save results"""
        
        # Augment the training dataset
        augmented_dataset = self.augment_training_dataset()
        
        # Save augmented dataset
        self.save_augmented_dataset(augmented_dataset)
        
        # Save summary
        self.save_augmentation_summary(augmented_dataset)
        
        print(f"\n🎉 Data augmentation complete!")
        print(f"💾 Files saved to: {self.output_directory}")


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Perform data augmentation on training dataset"
    )
    parser.add_argument(
        '--split-dataset-dir', '-s',
        default='split_dataset',
        help='Directory containing split datasets (default: split_dataset)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='augmented_dataset',
        help='Output directory for augmented dataset (default: augmented_dataset)'
    )
    parser.add_argument(
        '--random-seed', '-r',
        type=int,
        default=42,
        help='Random seed for reproducible augmentation (default: 42)'
    )
    
    # Individual augmentation control flags
    parser.add_argument(
        '--enable-z-offset',
        action='store_true',
        default=True,
        help='Enable Z-offset augmentation (default: enabled)'
    )
    parser.add_argument(
        '--disable-z-offset',
        action='store_true',
        help='Disable Z-offset augmentation'
    )
    parser.add_argument(
        '--enable-z-noise',
        action='store_true',
        default=True,
        help='Enable Z-noise augmentation (default: enabled)'
    )
    parser.add_argument(
        '--disable-z-noise',
        action='store_true',
        help='Disable Z-noise augmentation'
    )
    parser.add_argument(
        '--enable-x-shift',
        action='store_true',
        default=True,
        help='Enable X-shift augmentation (default: enabled)'
    )
    parser.add_argument(
        '--disable-x-shift',
        action='store_true',
        help='Disable X-shift augmentation'
    )
    parser.add_argument(
        '--enable-y-rotation',
        action='store_true',
        default=True,
        help='Enable Y-rotation augmentation (default: enabled)'
    )
    parser.add_argument(
        '--disable-y-rotation',
        action='store_true',
        help='Disable Y-rotation augmentation'
    )
    
    args = parser.parse_args()
    
    # Resolve augmentation flags (disable flags override enable flags)
    enable_z_offset = args.enable_z_offset and not args.disable_z_offset
    enable_z_noise = args.enable_z_noise and not args.disable_z_noise
    enable_x_shift = args.enable_x_shift and not args.disable_x_shift
    enable_y_rotation = args.enable_y_rotation and not args.disable_y_rotation
    
    print("DATA AUGMENTATION")
    print("=" * 50)
    print(f"Augmentation settings:")
    print(f"  Z-offset: {'✅ Enabled' if enable_z_offset else '❌ Disabled'}")
    print(f"  Z-noise:  {'✅ Enabled' if enable_z_noise else '❌ Disabled'}")
    print(f"  X-shift:  {'✅ Enabled' if enable_x_shift else '❌ Disabled'}")
    print(f"  Y-rotation: {'✅ Enabled' if enable_y_rotation else '❌ Disabled'}")
    
    try:
        # Create data augmentator
        augmentator = DataAugmentator(
            split_dataset_directory=args.split_dataset_dir,
            output_directory=args.output_dir,
            random_seed=args.random_seed,
            enable_z_offset=enable_z_offset,
            enable_x_shift=enable_x_shift,
            enable_y_rotation=enable_y_rotation,
            enable_z_noise=enable_z_noise
        )
        
        # Perform augmentation
        augmentator.augment_and_save()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())