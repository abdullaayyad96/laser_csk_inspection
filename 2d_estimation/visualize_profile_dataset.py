#!/usr/bin/env python3
"""
Profile Dataset Visualizer
--------------------------
Visualizes samples from the profile dataset showing:
- Hole number and dy value
- 2D scan profile (x vs z coordinates)
- Left and right countersink depths
- Interactive selection of profiles

Author: Abdulla Ayyad <abdullaayyad96@gmail.com>
Date: November 1, 2025
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import random


class ProfileDatasetVisualizer:
    def __init__(self, dataset_directory: str = "profile_dataset", dataset_name: str = "profile_dataset"):
        """
        Initialize the profile dataset visualizer
        
        Args:
            dataset_directory: Directory containing processed dataset files
            dataset_name: Base name for dataset files (without extension)
        """
        self.dataset_directory = Path(dataset_directory)
        self.dataset_name = dataset_name
        
        # Check if dataset files exist
        self.json_dataset_path = self.dataset_directory / f"{dataset_name}.json"
        self.pickle_dataset_path = self.dataset_directory / f"{dataset_name}.pkl"
        
        if not self.dataset_directory.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_directory}")
        
        print(f"📁 Dataset directory: {self.dataset_directory}")
        
        # Load dataset
        self.dataset = self.load_dataset()
        print(f"📊 Loaded {len(self.dataset)} profiles from dataset")

    def load_dataset(self) -> List[Dict]:
        """
        Load the processed dataset from JSON or pickle file
        
        Returns:
            List of processed profile data
        """
        # Try to load from pickle first (preserves numpy arrays)
        if self.pickle_dataset_path.exists():
            try:
                import pickle
                with open(self.pickle_dataset_path, 'rb') as f:
                    dataset = pickle.load(f)
                print(f"✅ Loaded dataset from pickle: {self.pickle_dataset_path.name}")
                return dataset
            except Exception as e:
                print(f"⚠️  Failed to load pickle file: {e}")
        
        # Fall back to JSON
        if self.json_dataset_path.exists():
            try:
                with open(self.json_dataset_path, 'r') as f:
                    dataset = json.load(f)
                
                # Convert lists back to numpy arrays
                for profile in dataset:
                    profile['x_array_mm'] = np.array(profile['x_array_mm'])
                    profile['y_array_mm'] = np.array(profile['y_array_mm'])
                    profile['z_array_mm'] = np.array(profile['z_array_mm'])
                
                print(f"✅ Loaded dataset from JSON: {self.json_dataset_path.name}")
                return dataset
            except Exception as e:
                print(f"❌ Failed to load JSON file: {e}")
        
        raise FileNotFoundError(f"No valid dataset files found in {self.dataset_directory}")

    def get_available_profiles(self) -> List[Dict]:
        """
        Get list of available profiles from the loaded dataset
        
        Returns:
            List of profile metadata
        """
        profiles = []
        for profile in self.dataset:
            profile_info = {
                'hole_number': profile['hole_number'],
                'hole_name': profile['hole_name'],
                'profile_id': profile['profile_id'],
                'dy_mm': profile['dy_mm'],
                'left_csk_depth_mm': profile['left_csk_depth_mm'],
                'right_csk_depth_mm': profile['right_csk_depth_mm'],
                'num_points': profile['num_points'],
                'x_array_mm': profile['x_array_mm'],
                'y_array_mm': profile['y_array_mm'],
                'z_array_mm': profile['z_array_mm']
            }
            profiles.append(profile_info)
        
        return profiles

    def visualize_profile(self, profile_info: Dict, show_plot: bool = True) -> None:
        """
        Visualize a single profile
        
        Args:
            profile_info: Profile data dictionary (from dataset)
            show_plot: Whether to display the plot
        """
        print(f"\n🎯 Visualizing Profile")
        print(f"=" * 50)
        print(f"Hole: {profile_info['hole_name']}")
        print(f"Profile ID: {profile_info['profile_id']}")
        print(f"DY: {profile_info['dy_mm']:.3f} mm")
        print(f"Left CSK Depth: {profile_info['left_csk_depth_mm']:.4f} mm")
        print(f"Right CSK Depth: {profile_info['right_csk_depth_mm']:.4f} mm")
        print(f"Points: {profile_info['num_points']:,}")
        
        # Get coordinate arrays (already in mm)
        x_array = profile_info['x_array_mm']
        y_array = profile_info['y_array_mm']
        z_array = profile_info['z_array_mm']
        
        if len(x_array) == 0:
            print("❌ No coordinate data available")
            return
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: X vs Z (main 2D scan profile)
        ax1.scatter(x_array, z_array, c='blue', s=2, alpha=0.7)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Z (mm)')
        ax1.set_title(f'{profile_info["hole_name"]} - Profile Scan (X vs Z)\nDY = {profile_info["dy_mm"]:.3f} mm')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Add depth annotations
        ax1.text(0.02, 0.98, f'Left CSK: {profile_info["left_csk_depth_mm"]:.4f} mm\nRight CSK: {profile_info["right_csk_depth_mm"]:.4f} mm',
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 2: Y vs Z (side view)
        ax2.scatter(y_array, z_array, c='red', s=2, alpha=0.7)
        ax2.set_xlabel('Y (mm)')
        ax2.set_ylabel('Z (mm)')
        ax2.set_title(f'{profile_info["hole_name"]} - Side View (Y vs Z)\nDY = {profile_info["dy_mm"]:.3f} mm')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        # Add statistics
        stats_text = f'Points: {len(x_array):,}\nX range: {np.min(x_array):.1f} to {np.max(x_array):.1f} mm\nY range: {np.min(y_array):.1f} to {np.max(y_array):.1f} mm\nZ range: {np.min(z_array):.1f} to {np.max(z_array):.1f} mm'
        ax2.text(0.02, 0.98, stats_text,
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return fig

    def list_available_profiles(self, profiles: List[Dict], max_display: int = 20) -> None:
        """List available profiles for selection"""
        print(f"\n📋 Available Profiles (showing first {min(max_display, len(profiles))}):")
        print("-" * 80)
        print(f"{'#':<3} {'Hole':<10} {'Profile':<12} {'DY (mm)':<8} {'Points':<7} {'Left CSK':<9} {'Right CSK':<10}")
        print("-" * 80)
        
        for i, profile in enumerate(profiles[:max_display]):
            print(f"{i:<3} {profile['hole_name']:<10} {profile['profile_id']:<12} "
                  f"{profile['dy_mm']:<8.3f} {profile['num_points']:<7,} "
                  f"{profile['left_csk_depth_mm']:<9.4f} {profile['right_csk_depth_mm']:<10.4f}")
        
        if len(profiles) > max_display:
            print(f"... and {len(profiles) - max_display} more profiles")

    def interactive_selection(self, profiles: List[Dict]) -> None:
        """Interactive profile selection and visualization"""
        while True:
            self.list_available_profiles(profiles)
            
            print(f"\n🎮 Options:")
            print(f"  Enter number (0-{len(profiles)-1}) to visualize a profile")
            print(f"  'r' or 'random' to show a random profile")
            print(f"  'q' or 'quit' to exit")
            print(f"  'h' or 'help' to show this help")
            
            try:
                choice = input("\nYour choice: ").strip().lower()
                
                if choice in ['q', 'quit']:
                    print("👋 Goodbye!")
                    break
                elif choice in ['h', 'help']:
                    continue
                elif choice in ['r', 'random']:
                    profile = random.choice(profiles)
                    self.visualize_profile(profile)
                else:
                    index = int(choice)
                    if 0 <= index < len(profiles):
                        profile = profiles[index]
                        self.visualize_profile(profile)
                    else:
                        print(f"❌ Invalid index. Please enter 0-{len(profiles)-1}")
                        
            except ValueError:
                print("❌ Invalid input. Please enter a number, 'r', or 'q'")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Visualize profiles from the dataset"
    )
    parser.add_argument(
        '--dataset-dir', '-d',
        default='profile_dataset',
        help='Directory containing processed dataset files (default: profile_dataset)'
    )
    parser.add_argument(
        '--dataset-name', '-n',
        default='profile_dataset',
        help='Base name for dataset files without extension (default: profile_dataset)'
    )
    parser.add_argument(
        '--profile-index', '-p',
        type=int,
        help='Specific profile index to visualize (skips interactive mode)'
    )
    parser.add_argument(
        '--random', '-r',
        action='store_true',
        help='Show a random profile (skips interactive mode)'
    )
    parser.add_argument(
        '--hole', '-H',
        help='Show profiles from specific hole number (e.g., 1167)'
    )
    
    args = parser.parse_args()
    
    print("PROFILE DATASET VISUALIZER")
    print("=" * 50)
    
    try:
        # Create visualizer
        visualizer = ProfileDatasetVisualizer(
            dataset_directory=args.dataset_dir,
            dataset_name=args.dataset_name
        )
        
        # Load available profiles
        profiles = visualizer.get_available_profiles()
        
        if not profiles:
            print("❌ No profiles found")
            return 1
        
        # Filter by hole if specified
        if args.hole:
            profiles = [p for p in profiles if p['hole_number'] == args.hole]
            if not profiles:
                print(f"❌ No profiles found for hole {args.hole}")
                return 1
            print(f"🎯 Filtered to {len(profiles)} profiles from Hole-{args.hole}")
        
        # Handle different modes
        if args.random:
            # Show random profile
            profile = random.choice(profiles)
            visualizer.visualize_profile(profile)
        elif args.profile_index is not None:
            # Show specific profile
            if 0 <= args.profile_index < len(profiles):
                profile = profiles[args.profile_index]
                visualizer.visualize_profile(profile)
            else:
                print(f"❌ Invalid profile index. Available: 0-{len(profiles)-1}")
                return 1
        else:
            # Interactive mode
            visualizer.interactive_selection(profiles)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())