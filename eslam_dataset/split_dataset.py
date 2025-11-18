#!/usr/bin/env python3
"""
Dataset Splitter
----------------
Randomly splits the profile dataset into training, validation, and testing sets.
Ensures proper stratification by hole numbers to maintain balanced representation.

Author: GitHub Copilot
Date: November 1, 2025
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random


class DatasetSplitter:
    def __init__(self, dataset_directory: str = "profile_dataset", 
                 output_directory: str = "split_dataset",
                 random_seed: int = 42,
                 splitting_mode: str = "hole_based"):
        """
        Initialize the dataset splitter
        
        Args:
            dataset_directory: Directory containing the processed dataset
            output_directory: Directory to save split datasets
            random_seed: Random seed for reproducible splits
            splitting_mode: Either "hole_based" or "profile_based"
        """
        self.dataset_directory = Path(dataset_directory)
        self.output_directory = Path(output_directory)
        self.random_seed = random_seed
        self.splitting_mode = splitting_mode
        
        # Validate splitting mode
        if splitting_mode not in ["hole_based", "profile_based"]:
            raise ValueError(f"Invalid splitting_mode: {splitting_mode}. Must be 'hole_based' or 'profile_based'")
        
        # Set random seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directory
        self.output_directory.mkdir(exist_ok=True)
        
        print(f"📁 Dataset directory: {self.dataset_directory}")
        print(f"💾 Output directory: {self.output_directory}")
        print(f"🎲 Random seed: {random_seed}")
        print(f"🔄 Splitting mode: {splitting_mode}")
        
        # Load dataset
        self.dataset = self.load_dataset()
        print(f"📊 Loaded {len(self.dataset)} profiles from dataset")

    def load_dataset(self) -> List[Dict]:
        """Load the processed dataset from pickle or JSON file"""
        # Try to load from pickle first (preserves numpy arrays)
        pickle_path = self.dataset_directory / "profile_dataset.pkl"
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    dataset = pickle.load(f)
                print(f"✅ Loaded dataset from pickle: {pickle_path.name}")
                return dataset
            except Exception as e:
                print(f"⚠️  Failed to load pickle file: {e}")
        
        # Fall back to JSON
        json_path = self.dataset_directory / "profile_dataset.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    dataset = json.load(f)
                
                # Convert lists back to numpy arrays
                for profile in dataset:
                    profile['x_array_mm'] = np.array(profile['x_array_mm'])
                    profile['y_array_mm'] = np.array(profile['y_array_mm'])
                    profile['z_array_mm'] = np.array(profile['z_array_mm'])
                
                print(f"✅ Loaded dataset from JSON: {json_path.name}")
                return dataset
            except Exception as e:
                print(f"❌ Failed to load JSON file: {e}")
        
        raise FileNotFoundError(f"No valid dataset files found in {self.dataset_directory}")

    def group_profiles_by_hole(self) -> Dict[str, List[Dict]]:
        """Group profiles by hole number for stratified splitting"""
        hole_groups = defaultdict(list)
        
        for profile in self.dataset:
            hole_number = profile['hole_number']
            hole_groups[hole_number].append(profile)
        
        print(f"🎯 Grouped profiles by {len(hole_groups)} unique holes:")
        for hole_num, profiles in sorted(hole_groups.items()):
            print(f"  Hole-{hole_num}: {len(profiles)} profiles")
        
        return dict(hole_groups)

    def split_holes_stratified(self, hole_groups: Dict[str, List[Dict]], 
                              train_ratio: float = 0.7, 
                              val_ratio: float = 0.15, 
                              test_ratio: float = 0.15) -> Tuple[List[str], List[str], List[str]]:
        """
        Split holes into train/val/test sets with stratification
        
        Args:
            hole_groups: Dictionary mapping hole numbers to their profiles
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            
        Returns:
            Tuple of (train_holes, val_holes, test_holes)
        """
        # Verify ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        hole_numbers = list(hole_groups.keys())
        random.shuffle(hole_numbers)  # Shuffle holes randomly
        
        n_holes = len(hole_numbers)
        n_train = int(n_holes * train_ratio)
        n_val = int(n_holes * val_ratio)
        n_test = n_holes - n_train - n_val  # Ensure all holes are assigned
        
        train_holes = hole_numbers[:n_train]
        val_holes = hole_numbers[n_train:n_train + n_val]
        test_holes = hole_numbers[n_train + n_val:]
        
        print(f"\n📊 Hole distribution:")
        print(f"  Training: {len(train_holes)} holes ({len(train_holes)/n_holes*100:.1f}%)")
        print(f"  Validation: {len(val_holes)} holes ({len(val_holes)/n_holes*100:.1f}%)")
        print(f"  Testing: {len(test_holes)} holes ({len(test_holes)/n_holes*100:.1f}%)")
        
        return train_holes, val_holes, test_holes

    def split_profiles_random(self, train_ratio: float = 0.7, 
                             val_ratio: float = 0.15, 
                             test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split profiles randomly (ignoring hole assignments)
        
        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Verify ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # Make a copy of dataset and shuffle it
        dataset_copy = self.dataset.copy()
        random.shuffle(dataset_copy)
        
        n_profiles = len(dataset_copy)
        n_train = int(n_profiles * train_ratio)
        n_val = int(n_profiles * val_ratio)
        n_test = n_profiles - n_train - n_val  # Ensure all profiles are assigned
        
        train_dataset = dataset_copy[:n_train]
        val_dataset = dataset_copy[n_train:n_train + n_val]
        test_dataset = dataset_copy[n_train + n_val:]
        
        print(f"\n📊 Profile distribution (random split):")
        print(f"  Training: {len(train_dataset)} profiles ({len(train_dataset)/n_profiles*100:.1f}%)")
        print(f"  Validation: {len(val_dataset)} profiles ({len(val_dataset)/n_profiles*100:.1f}%)")
        print(f"  Testing: {len(test_dataset)} profiles ({len(test_dataset)/n_profiles*100:.1f}%)")
        
        return train_dataset, val_dataset, test_dataset

    def create_split_datasets(self, hole_groups: Dict[str, List[Dict]], 
                             train_holes: List[str], 
                             val_holes: List[str], 
                             test_holes: List[str]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/val/test datasets from hole assignments"""
        
        train_dataset = []
        val_dataset = []
        test_dataset = []
        
        # Assign profiles based on hole assignments
        for hole_num, profiles in hole_groups.items():
            if hole_num in train_holes:
                train_dataset.extend(profiles)
            elif hole_num in val_holes:
                val_dataset.extend(profiles)
            elif hole_num in test_holes:
                test_dataset.extend(profiles)
        
        # Shuffle within each dataset
        random.shuffle(train_dataset)
        random.shuffle(val_dataset)
        random.shuffle(test_dataset)
        
        print(f"\n📈 Profile distribution:")
        print(f"  Training: {len(train_dataset)} profiles ({len(train_dataset)/len(self.dataset)*100:.1f}%)")
        print(f"  Validation: {len(val_dataset)} profiles ({len(val_dataset)/len(self.dataset)*100:.1f}%)")
        print(f"  Testing: {len(test_dataset)} profiles ({len(test_dataset)/len(self.dataset)*100:.1f}%)")
        
        return train_dataset, val_dataset, test_dataset

    def save_split_datasets(self, train_dataset: List[Dict], 
                           val_dataset: List[Dict], 
                           test_dataset: List[Dict]) -> None:
        """Save the split datasets in both pickle and JSON formats"""
        
        datasets = {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }
        
        for split_name, dataset in datasets.items():
            print(f"\n💾 Saving {split_name} dataset ({len(dataset)} profiles)...")
            
            # Save as pickle (preserves numpy arrays)
            pickle_file = self.output_directory / f"{split_name}_dataset.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"  🥒 Saved pickle: {pickle_file.name}")
            
            # Save as JSON (convert numpy arrays to lists)
            json_data = []
            for profile in dataset:
                json_profile = profile.copy()
                json_profile['x_array_mm'] = profile['x_array_mm'].tolist()
                json_profile['y_array_mm'] = profile['y_array_mm'].tolist()
                json_profile['z_array_mm'] = profile['z_array_mm'].tolist()
                json_data.append(json_profile)
            
            json_file = self.output_directory / f"{split_name}_dataset.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"  📄 Saved JSON: {json_file.name}")

    def save_split_summary(self, train_dataset: List[Dict], 
                          val_dataset: List[Dict], 
                          test_dataset: List[Dict],
                          hole_groups: Dict[str, List[Dict]] = None, 
                          train_holes: List[str] = None, 
                          val_holes: List[str] = None, 
                          test_holes: List[str] = None) -> None:
        """Save summary of the dataset split"""
        
        summary_file = self.output_directory / "split_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("DATASET SPLIT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Splitting mode: {self.splitting_mode}\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Total profiles: {len(self.dataset)}\n")
            if hole_groups:
                f.write(f"Total holes: {len(hole_groups)}\n")
            f.write("\n")
            
            # Split statistics
            f.write("SPLIT STATISTICS\n")
            f.write("-" * 20 + "\n")
            if self.splitting_mode == "hole_based" and train_holes is not None:
                f.write(f"Training:   {len(train_dataset):4d} profiles ({len(train_dataset)/len(self.dataset)*100:5.1f}%) from {len(train_holes):2d} holes\n")
                f.write(f"Validation: {len(val_dataset):4d} profiles ({len(val_dataset)/len(self.dataset)*100:5.1f}%) from {len(val_holes):2d} holes\n")
                f.write(f"Testing:    {len(test_dataset):4d} profiles ({len(test_dataset)/len(self.dataset)*100:5.1f}%) from {len(test_holes):2d} holes\n\n")
            else:
                f.write(f"Training:   {len(train_dataset):4d} profiles ({len(train_dataset)/len(self.dataset)*100:5.1f}%)\n")
                f.write(f"Validation: {len(val_dataset):4d} profiles ({len(val_dataset)/len(self.dataset)*100:5.1f}%)\n")
                f.write(f"Testing:    {len(test_dataset):4d} profiles ({len(test_dataset)/len(self.dataset)*100:5.1f}%)\n\n")
            
            # Hole assignments (only for hole-based splitting)
            if self.splitting_mode == "hole_based" and train_holes is not None:
                f.write("HOLE ASSIGNMENTS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Training holes:   {sorted(train_holes)}\n")
                f.write(f"Validation holes: {sorted(val_holes)}\n")
                f.write(f"Testing holes:    {sorted(test_holes)}\n\n")
            
            # Detailed breakdown
            f.write("DETAILED BREAKDOWN\n")
            f.write("-" * 20 + "\n")
            
            if self.splitting_mode == "hole_based" and train_holes is not None:
                splits = [
                    ("Training", train_holes, train_dataset),
                    ("Validation", val_holes, val_dataset),
                    ("Testing", test_holes, test_dataset)
                ]
                
                for split_name, holes, dataset in splits:
                    f.write(f"\n{split_name}:\n")
                    hole_profile_counts = {}
                    for profile in dataset:
                        hole = profile['hole_number']
                        hole_profile_counts[hole] = hole_profile_counts.get(hole, 0) + 1
                    
                    for hole in sorted(holes):
                        count = hole_profile_counts.get(hole, 0)
                        left_depth = next((p['left_csk_depth_mm'] for p in dataset if p['hole_number'] == hole), 'N/A')
                        right_depth = next((p['right_csk_depth_mm'] for p in dataset if p['hole_number'] == hole), 'N/A')
                        f.write(f"  Hole-{hole}: {count:3d} profiles (L:{left_depth:.3f}, R:{right_depth:.3f})\n")
            else:
                # For profile-based splitting, show hole distribution in each split
                splits = [
                    ("Training", train_dataset),
                    ("Validation", val_dataset),
                    ("Testing", test_dataset)
                ]
                
                for split_name, dataset in splits:
                    f.write(f"\n{split_name}:\n")
                    hole_profile_counts = {}
                    for profile in dataset:
                        hole = profile['hole_number']
                        hole_profile_counts[hole] = hole_profile_counts.get(hole, 0) + 1
                    
                    for hole in sorted(hole_profile_counts.keys()):
                        count = hole_profile_counts[hole]
                        f.write(f"  Hole-{hole}: {count:3d} profiles\n")
        
        print(f"📋 Saved split summary: {summary_file.name}")

    def split_dataset(self, train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, 
                     test_ratio: float = 0.15) -> None:
        """
        Main method to split the dataset
        
        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
        """
        print(f"\n🚀 Starting dataset split...")
        print(f"📊 Split ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
        
        if self.splitting_mode == "hole_based":
            print("🎯 Using hole-based stratified splitting (no data leakage)")
            
            # Group profiles by hole
            hole_groups = self.group_profiles_by_hole()
            
            # Split holes into train/val/test
            train_holes, val_holes, test_holes = self.split_holes_stratified(
                hole_groups, train_ratio, val_ratio, test_ratio
            )
            
            # Create split datasets
            train_dataset, val_dataset, test_dataset = self.create_split_datasets(
                hole_groups, train_holes, val_holes, test_holes
            )
            
            # Save summary with hole information
            hole_groups_for_summary = hole_groups
            train_holes_for_summary = train_holes
            val_holes_for_summary = val_holes
            test_holes_for_summary = test_holes
            
        else:  # profile_based
            print("🎲 Using random profile-based splitting (faster training)")
            
            # Split profiles randomly
            train_dataset, val_dataset, test_dataset = self.split_profiles_random(
                train_ratio, val_ratio, test_ratio
            )
            
            # No hole information for summary
            hole_groups_for_summary = None
            train_holes_for_summary = None
            val_holes_for_summary = None
            test_holes_for_summary = None
        
        # Save split datasets
        self.save_split_datasets(train_dataset, val_dataset, test_dataset)
        
        # Save summary
        self.save_split_summary(
            train_dataset, val_dataset, test_dataset,
            hole_groups_for_summary, train_holes_for_summary, 
            val_holes_for_summary, test_holes_for_summary
        )
        
        print(f"\n🎉 Dataset split complete!")
        print(f"💾 Files saved to: {self.output_directory}")


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split profile dataset into training, validation, and testing sets"
    )
    parser.add_argument(
        '--dataset-dir', '-d',
        default='profile_dataset',
        help='Directory containing the processed dataset (default: profile_dataset)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='split_dataset',
        help='Output directory for split datasets (default: split_dataset)'
    )
    parser.add_argument(
        '--train-ratio', '-t',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio', '-v',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio', '-e',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--random-seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    parser.add_argument(
        '--splitting-mode', '-m',
        choices=['hole_based', 'profile_based'],
        default='hole_based',
        help='Splitting strategy: hole_based (stratified, no leakage) or profile_based (random profiles) (default: hole_based)'
    )
    
    args = parser.parse_args()
    
    print("DATASET SPLITTER")
    print("=" * 50)
    
    try:
        # Create dataset splitter
        splitter = DatasetSplitter(
            dataset_directory=args.dataset_dir,
            output_directory=args.output_dir,
            random_seed=args.random_seed,
            splitting_mode=args.splitting_mode
        )
        
        # Split dataset
        splitter.split_dataset(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())