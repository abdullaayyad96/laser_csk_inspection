#!/usr/bin/env python3
"""
PCD Loading and Splitting Test Script
------------------------------------
Tests the robust PCD loading functionality and visualizes bilateral splitting.

Features:
1. Load PCD file with infinite value filtering
2. Visualize original point cloud
3. Split point cloud into left/right halves
4. Visualize each half separately
5. Save each half as separate PCD files
6. Display statistics and filtering results

Usage:
    python test_pcd_loading_and_splitting.py <pcd_file> [--split-method center|median]
    
Requirements:
    pip install numpy matplotlib open3d (optional)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from typing import Tuple, Dict

# Import common point cloud utilities - implementing inline to avoid dependency
# try:
#     from point_cloud_utils import (
#         PointCloudLoader, PointCloudAnalyzer, PointCloudSaver, PointCloudVisualizer
#     )
# except ImportError:
#     print("Error: point_cloud_utils.py not found in current directory")
#     print("Please ensure point_cloud_utils.py is in the same directory")
#     sys.exit(1)

class PCDTester:
    def __init__(self, split_method: str = 'center', no_use_zone_percent: float = 0.0):
        """
        Initialize PCD tester
        
        Args:
            split_method: 'center' or 'median' for splitting point cloud
            no_use_zone_percent: Percentage of middle area to exclude (0-50)
        """
        self.split_method = split_method.lower()
        if self.split_method not in ['center', 'median']:
            raise ValueError("split_method must be 'center' or 'median'")
        
        self.no_use_zone_percent = no_use_zone_percent
        if not (0 <= no_use_zone_percent <= 50):
            raise ValueError("no_use_zone_percent must be between 0 and 50")
        
        print(f"🔧 PCD Tester Configuration:")
        print(f"   Split method: {self.split_method}")
        print(f"   No-use zone: {self.no_use_zone_percent}% of middle area")
    
    def load_pcd_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load PCD file with robust infinite value filtering
        
        Args:
            filename: Path to PCD file
            
        Returns:
            Tuple of (x, y, z) coordinate arrays
        """
        print(f"📂 Loading PCD file: {filename}")
        
        # Read PCD file
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Find the start of data
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('DATA'):
                data_start = i + 1
                break
        
        # Extract point data
        points = []
        for line in lines[data_start:]:
            if line.strip():
                try:
                    coords = [float(x) for x in line.strip().split()[:3]]
                    points.append(coords)
                except ValueError:
                    continue
        
        points = np.array(points)
        
        if len(points) == 0:
            print("❌ No valid points found in file")
            return np.array([]), np.array([]), np.array([])
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Filter infinite values
        finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x_clean = x[finite_mask]
        y_clean = y[finite_mask]
        z_clean = z[finite_mask]
        
        removed_count = len(x) - len(x_clean)
        if removed_count > 0:
            print(f"🧹 Removed {removed_count} points with infinite values")
        
        print(f"✅ Loaded {len(x_clean)} valid points")
        return x_clean, y_clean, z_clean
    
    def analyze_point_cloud(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
        """
        Analyze point cloud statistics
        
        Args:
            x, y, z: Point cloud coordinates
            
        Returns:
            Dictionary with analysis results
        """
        if len(x) == 0:
            return {'count': 0}
        
        analysis = {
            'count': len(x),
            'x_range': [np.min(x), np.max(x)],
            'y_range': [np.min(y), np.max(y)],
            'z_range': [np.min(z), np.max(z)],
            'x_span': np.max(x) - np.min(x),
            'y_span': np.max(y) - np.min(y),
            'z_span': np.max(z) - np.min(z),
            'centroid': [np.mean(x), np.mean(y), np.mean(z)]
        }
        
        # Print analysis
        print(f"📊 Point Cloud Analysis:")
        print(f"   Total points: {analysis['count']}")
        print(f"   X range: [{analysis['x_range'][0]:.3f}, {analysis['x_range'][1]:.3f}] span: {analysis['x_span']:.3f}")
        print(f"   Y range: [{analysis['y_range'][0]:.3f}, {analysis['y_range'][1]:.3f}] span: {analysis['y_span']:.3f}")
        print(f"   Z range: [{analysis['z_range'][0]:.3f}, {analysis['z_range'][1]:.3f}] span: {analysis['z_span']:.3f}")
        print(f"   Centroid: ({analysis['centroid'][0]:.3f}, {analysis['centroid'][1]:.3f}, {analysis['centroid'][2]:.3f})")
        
        return analysis
    
    def split_point_cloud(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[Dict, Dict, float]:
        """
        Split point cloud into left and right halves with optional no-use zone
        
        Args:
            x, y, z: Point cloud coordinates
            
        Returns:
            left_points: Dictionary with left half points
            right_points: Dictionary with right half points
            split_line: X-coordinate of the division line
        """
        if len(x) == 0:
            empty_dict = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
            return empty_dict, empty_dict, 0.0
        
        # Determine split position based on X coordinates
        if self.split_method == 'center':
            split_line = (np.min(x) + np.max(x)) / 2
        else:  # median
            split_line = np.median(x)
        
        # Calculate no-use zone boundaries
        if self.no_use_zone_percent > 0:
            x_range = np.max(x) - np.min(x)
            half_zone_width = (x_range * self.no_use_zone_percent / 100) / 2
            no_use_lower = split_line - half_zone_width
            no_use_upper = split_line + half_zone_width
            
            # Create mask for usable points (excluding no-use zone)
            use_mask = (x < no_use_lower) | (x > no_use_upper)
            
            # Apply mask to get usable points
            usable_x = x[use_mask]
            usable_y = y[use_mask]
            usable_z = z[use_mask]
            
            excluded_count = len(x) - len(usable_x)
            
            # Split the usable points based on X coordinates
            left_mask = usable_x < split_line
            right_mask = usable_x > split_line
            
            left_points = {
                'x': usable_x[left_mask],
                'y': usable_y[left_mask], 
                'z': usable_z[left_mask]
            }
            
            right_points = {
                'x': usable_x[right_mask],
                'y': usable_y[right_mask],
                'z': usable_z[right_mask]
            }
            
            print(f"🚫 No-use zone applied:")
            print(f"   Zone bounds: {no_use_lower:.3f} to {no_use_upper:.3f}")
            print(f"   Excluded points: {excluded_count}")
            print(f"   Usable points: {len(usable_x)}")
            
        else:
            # No exclusion zone - normal splitting based on X coordinates
            left_mask = x < split_line
            right_mask = x > split_line
            
            left_points = {
                'x': x[left_mask],
                'y': y[left_mask],
                'z': z[left_mask]
            }
            
            right_points = {
                'x': x[right_mask],
                'y': y[right_mask],
                'z': z[right_mask]
            }
        
        # Print split information
        print(f"📊 Point cloud split results:")
        print(f"   Split method: {self.split_method}")
        print(f"   Split line (x): {split_line:.3f}")
        print(f"   Left points: {len(left_points['x'])}")
        print(f"   Right points: {len(right_points['x'])}")
        print(f"   Total original: {len(x)}")
        
        return left_points, right_points, split_line
    
    def save_pcd_half(self, points: Dict, filename: str, side: str):
        """
        Save half of point cloud as PCD file
        
        Args:
            points: Dictionary with x, y, z coordinates
            filename: Output filename
            side: 'left' or 'right' for identification
        """
        x, y, z = points['x'], points['y'], points['z']
        num_points = len(x)
        
        print(f"💾 Saving {side} half: {num_points} points to {filename}")
        
        with open(filename, 'w') as f:
            # Write PCD header
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write(f"WIDTH {num_points}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\n")
            f.write("DATA ascii\n")
            
            # Write point data
            for i in range(num_points):
                f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f}\n")
    
    def visualize_point_clouds(self, original_data: Tuple, left_points: Dict, right_points: Dict, 
                             split_line: float, base_filename: str):
        """
        Create comprehensive visualization of original and split point clouds
        
        Args:
            original_data: Tuple of (x, y, z) original coordinates
            left_points: Dictionary with left half points
            right_points: Dictionary with right half points
            split_line: X-coordinate of the division line
            base_filename: Base filename for saving plots
        """
        x_orig, y_orig, z_orig = original_data
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Original point cloud (top view)
        ax1 = fig.add_subplot(2, 3, 1)
        scatter1 = ax1.scatter(x_orig, y_orig, c=z_orig, cmap='viridis', s=1, alpha=0.6)
        ax1.axvline(x=split_line, color='red', linestyle='--', linewidth=2, label=f'Split line (x={split_line:.3f})')
        
        # Add no-use zone visualization if enabled
        if self.no_use_zone_percent > 0:
            x_range = np.max(x_orig) - np.min(x_orig)
            half_zone_width = (x_range * self.no_use_zone_percent / 100) / 2
            no_use_lower = split_line - half_zone_width
            no_use_upper = split_line + half_zone_width
            ax1.axvspan(no_use_lower, no_use_upper, alpha=0.3, color='red', 
                       label=f'No-use zone ({self.no_use_zone_percent}%)')
        
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Original Point Cloud (Top View)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Z Coordinate')
        
        # Left half (top view)
        ax2 = fig.add_subplot(2, 3, 2)
        if len(left_points['x']) > 0:
            scatter2 = ax2.scatter(left_points['x'], left_points['y'], c=left_points['z'], 
                                 cmap='viridis', s=1, alpha=0.6)
            plt.colorbar(scatter2, ax=ax2, label='Z Coordinate')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title(f'Left Half ({len(left_points["x"])} points)')
        ax2.grid(True, alpha=0.3)
        
        # Right half (top view)
        ax3 = fig.add_subplot(2, 3, 3)
        if len(right_points['x']) > 0:
            scatter3 = ax3.scatter(right_points['x'], right_points['y'], c=right_points['z'], 
                                 cmap='viridis', s=1, alpha=0.6)
            plt.colorbar(scatter3, ax=ax3, label='Z Coordinate')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title(f'Right Half ({len(right_points["x"])} points)')
        ax3.grid(True, alpha=0.3)
        
        # 3D view of original
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        if len(x_orig) > 0:
            ax4.scatter(x_orig, y_orig, z_orig, c=z_orig, cmap='viridis', s=1, alpha=0.6)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.set_title('Original 3D View')
        
        # 3D view of left half
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        if len(left_points['x']) > 0:
            ax5.scatter(left_points['x'], left_points['y'], left_points['z'], 
                       c='blue', s=1, alpha=0.6)
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        ax5.set_title('Left Half 3D')
        
        # 3D view of right half
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        if len(right_points['x']) > 0:
            ax6.scatter(right_points['x'], right_points['y'], right_points['z'], 
                       c='red', s=1, alpha=0.6)
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        ax6.set_title('Right Half 3D')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"{base_filename}_bilateral_split.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved as: {plot_filename}")
        
        plt.show()
        return plot_filename
    
    def test_pcd_file(self, filename: str):
        """
        Complete test of PCD loading and splitting
        
        Args:
            filename: Path to PCD file
        """
        print(f"🚀 Starting PCD Loading and Splitting Test")
        print(f"{'='*60}")
        
        # Get base filename for output files
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        
        try:
            # 1. Load point cloud
            x, y, z = self.load_pcd_file(filename)
            
            # 2. Analyze original point cloud
            analysis = self.analyze_point_cloud(x, y, z)
            
            # 3. Split point cloud
            left_points, right_points, split_line = self.split_point_cloud(x, y, z)
            
            # 4. Save split halves
            left_filename = f"{base_filename}_left.pcd"
            right_filename = f"{base_filename}_right.pcd"
            
            self.save_pcd_half(left_points, left_filename, 'left')
            self.save_pcd_half(right_points, right_filename, 'right')
            
            # 5. Create visualization
            self.visualize_point_clouds((x, y, z), left_points, right_points, split_line, base_filename)
            
            print(f"\n✅ Test completed successfully!")
            print(f"📁 Output files:")
            print(f"   - {left_filename}")
            print(f"   - {right_filename}")
            print(f"   - {base_filename}_splitting_analysis.png")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PCD Loading and Splitting Tester with No-Use Zone')
    parser.add_argument('filename', help='Path to PCD file')
    parser.add_argument('--split-method', choices=['center', 'median'], default='center',
                       help='Method for splitting point cloud (default: center)')
    parser.add_argument('--no-use-zone', type=float, default=30.0, 
                       help='Percentage of middle area to exclude (0-50, default: 0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"❌ File not found: {args.filename}")
        return 1
    
    if not (0 <= args.no_use_zone <= 50):
        print(f"❌ Invalid no-use zone percentage: {args.no_use_zone}. Must be between 0 and 50.")
        return 1
    
    # Create tester with no-use zone
    tester = PCDTester(split_method=args.split_method, no_use_zone_percent=args.no_use_zone)
    
    try:
        # Run test
        tester.test_pcd_file(args.filename)
        return 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())