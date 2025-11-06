#!/usr/bin/env python3
"""
Point Cloud Utilities
--------------------
Common utilities for loading, processing, and visualizing point clouds.
Shared by bilateral_countersink_estimator.py and test_pcd_loading_and_splitting.py

Features:
- Robust point cloud loading with infinite value filtering
- Support for PCD, PLY, XYZ, TXT, CSV formats
- Point cloud analysis and statistics
- Bilateral splitting functionality
- Comprehensive error handling and reporting

Usage:
    from point_cloud_utils import PointCloudLoader, PointCloudAnalyzer
"""

import numpy as np
import os
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PointCloudLoader:
    """
    Robust point cloud loader with support for multiple formats and infinite value filtering
    """
    
    @staticmethod
    def load_point_cloud(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load point cloud from various file formats with robust error handling
        
        Args:
            filename: Path to point cloud file
            
        Returns:
            Tuple of (x, y, z) coordinate arrays
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        file_ext = os.path.splitext(filename)[1].lower()
        print(f"📂 Loading point cloud: {filename} (format: {file_ext})")
        
        # Method 1: Try Open3D (supports many formats including PCD, PLY, XYZ, etc.)
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(filename)
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                raise ValueError("Point cloud is empty")
            
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            print(f"✅ Loaded {len(points)} points using Open3D")
            return x, y, z
            
        except ImportError:
            print("⚠️  Open3D not available, trying alternative methods...")
        except Exception as e:
            print(f"⚠️  Open3D failed: {e}, trying alternative methods...")
        
        # Method 2: Handle PCD files manually (ASCII format)
        if file_ext == '.pcd':
            try:
                x, y, z = PointCloudLoader._load_pcd_file(filename)
                print(f"✅ Loaded {len(x)} points from PCD file (manual parsing)")
                return x, y, z
            except Exception as e:
                print(f"⚠️  Manual PCD parsing failed: {e}")
        
        # Method 3: Handle PLY files manually (ASCII format)
        elif file_ext == '.ply':
            try:
                x, y, z = PointCloudLoader._load_ply_file(filename)
                print(f"✅ Loaded {len(x)} points from PLY file (manual parsing)")
                return x, y, z
            except Exception as e:
                print(f"⚠️  Manual PLY parsing failed: {e}")
        
        # Method 4: Simple text file (XYZ, TXT, CSV)
        elif file_ext in ['.xyz', '.txt', '.csv']:
            try:
                x, y, z = PointCloudLoader._load_text_file(filename)
                print(f"✅ Loaded {len(x)} points from text file")
                return x, y, z
            except Exception as e:
                print(f"⚠️  Text file loading failed: {e}")
        
        # Method 5: Last resort - try to auto-detect format
        print("🔄 Attempting auto-detection...")
        
        # Try each method in order
        methods = [
            ('PCD (manual)', PointCloudLoader._load_pcd_file),
            ('PLY (manual)', PointCloudLoader._load_ply_file),
            ('Text file', PointCloudLoader._load_text_file)
        ]
        
        for method_name, method_func in methods:
            try:
                x, y, z = method_func(filename)
                print(f"✅ Auto-detection successful: {method_name} - {len(x)} points")
                return x, y, z
            except Exception as e:
                print(f"⚠️  {method_name} failed: {e}")
        
        # If all methods fail
        raise ValueError(f"Could not load point cloud from {filename}. "
                        f"Supported formats: PCD, PLY, XYZ, TXT, CSV. "
                        f"Please ensure the file is valid and contains x,y,z coordinates.")
    
    @staticmethod
    def _load_pcd_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load ASCII PCD file manually with infinite value filtering"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        data_start = 0
        points_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('POINTS'):
                points_count = int(line.split()[1])
            elif line.startswith('DATA'):
                if 'ascii' not in line.lower():
                    raise ValueError("Only ASCII PCD files are supported")
                data_start = i + 1
                break
        
        if data_start == 0:
            raise ValueError("Could not find DATA section in PCD file")
        
        # Read point data with filtering
        point_data = []
        invalid_points = 0
        inf_points = 0
        nan_points = 0
        parsing_errors = 0
        
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if line:
                values = line.split()
                if len(values) >= 3:
                    try:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        
                        # Check for NaN values
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            nan_points += 1
                            continue
                        
                        # Check for infinite values
                        if np.isinf(x) or np.isinf(y) or np.isinf(z):
                            inf_points += 1
                            continue
                        
                        # Check for other invalid values
                        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                            invalid_points += 1
                            continue
                        
                        point_data.append([x, y, z])
                        
                    except ValueError:
                        parsing_errors += 1
                        continue  # Skip invalid lines
        
        if len(point_data) == 0:
            raise ValueError("No valid point data found in PCD file")
        
        # Report filtering results
        total_filtered = invalid_points + inf_points + nan_points + parsing_errors
        if total_filtered > 0:
            print(f"⚠️  Filtered out {total_filtered} invalid points:")
            if inf_points > 0:
                print(f"   - {inf_points} points with infinite values")
            if nan_points > 0:
                print(f"   - {nan_points} points with NaN values")
            if invalid_points > 0:
                print(f"   - {invalid_points} points with other invalid values")
            if parsing_errors > 0:
                print(f"   - {parsing_errors} lines with parsing errors")
            print(f"   - Kept {len(point_data)} valid points")
        
        points = np.array(point_data)
        return points[:, 0], points[:, 1], points[:, 2]
    
    @staticmethod
    def _load_ply_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load ASCII PLY file manually with infinite value filtering"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        data_start = 0
        vertex_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line == 'end_header':
                data_start = i + 1
                break
        
        if data_start == 0:
            raise ValueError("Could not find end_header in PLY file")
        
        # Read vertex data
        point_data = []
        invalid_points = 0
        inf_points = 0
        nan_points = 0
        parsing_errors = 0
        
        for i in range(data_start, min(data_start + vertex_count, len(lines))):
            line = lines[i].strip()
            if line:
                values = line.split()
                if len(values) >= 3:
                    try:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        
                        # Check for NaN values
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            nan_points += 1
                            continue
                        
                        # Check for infinite values
                        if np.isinf(x) or np.isinf(y) or np.isinf(z):
                            inf_points += 1
                            continue
                        
                        # Check for other invalid values
                        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                            invalid_points += 1
                            continue
                        
                        point_data.append([x, y, z])
                        
                    except ValueError:
                        parsing_errors += 1
                        continue  # Skip invalid lines
        
        if len(point_data) == 0:
            raise ValueError("No valid vertex data found in PLY file")
        
        # Report filtering results
        total_filtered = invalid_points + inf_points + nan_points + parsing_errors
        if total_filtered > 0:
            print(f"⚠️  Filtered out {total_filtered} invalid points from PLY:")
            if inf_points > 0:
                print(f"   - {inf_points} points with infinite values")
            if nan_points > 0:
                print(f"   - {nan_points} points with NaN values")
            if invalid_points > 0:
                print(f"   - {invalid_points} points with other invalid values")
            if parsing_errors > 0:
                print(f"   - {parsing_errors} lines with parsing errors")
            print(f"   - Kept {len(point_data)} valid points")
        
        points = np.array(point_data)
        return points[:, 0], points[:, 1], points[:, 2]
    
    @staticmethod
    def _load_text_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load simple text file (XYZ, TXT, CSV) with infinite value filtering"""
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Determine delimiter
        delimiter = None
        if file_ext == '.csv':
            delimiter = ','
        
        # Try to load with numpy
        try:
            if delimiter:
                data = np.loadtxt(filename, delimiter=delimiter)
            else:
                data = np.loadtxt(filename)
        except ValueError as e:
            # If numpy fails, try pandas for more robust CSV handling
            try:
                import pandas as pd
                df = pd.read_csv(filename, sep=delimiter, header=None, comment='#')
                data = df.values
            except ImportError:
                raise ValueError(f"Could not parse text file: {e}")
            except Exception as e:
                raise ValueError(f"Could not parse text file with pandas: {e}")
        
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"File must have at least 3 columns (x,y,z), got shape {data.shape}")
        
        # Filter out infinite or NaN values
        invalid_points = 0
        inf_points = 0
        nan_points = 0
        
        # Count invalid points
        finite_mask = np.isfinite(data[:, :3])  # Check first 3 columns
        valid_rows = np.all(finite_mask, axis=1)
        
        invalid_data = data[~valid_rows]
        for row in invalid_data:
            if np.any(np.isinf(row[:3])):
                inf_points += 1
            elif np.any(np.isnan(row[:3])):
                nan_points += 1
            else:
                invalid_points += 1
        
        # Filter to valid rows
        valid_data = data[valid_rows]
        
        # Report filtering results
        total_filtered = invalid_points + inf_points + nan_points
        if total_filtered > 0:
            print(f"⚠️  Filtered out {total_filtered} invalid points from text file:")
            if inf_points > 0:
                print(f"   - {inf_points} points with infinite values")
            if nan_points > 0:
                print(f"   - {nan_points} points with NaN values")
            if invalid_points > 0:
                print(f"   - {invalid_points} points with other invalid values")
            print(f"   - Kept {len(valid_data)} valid points")
        
        if len(valid_data) == 0:
            raise ValueError("No valid points remaining after filtering infinite/NaN values")
        
        return valid_data[:, 0], valid_data[:, 1], valid_data[:, 2]


class PointCloudAnalyzer:
    """
    Point cloud analysis and processing utilities
    """
    
    @staticmethod
    def analyze_point_cloud(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
        """
        Analyze point cloud statistics
        
        Args:
            x, y, z: Point cloud coordinates
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'total_points': len(x),
            'x_range': (np.min(x), np.max(x)),
            'y_range': (np.min(y), np.max(y)),
            'z_range': (np.min(z), np.max(z)),
            'x_center': (np.min(x) + np.max(x)) / 2,
            'y_center': (np.min(y) + np.max(y)) / 2,
            'z_center': (np.min(z) + np.max(z)) / 2,
            'x_median': np.median(x),
            'y_median': np.median(y),
            'z_median': np.median(z)
        }
        
        # Calculate bounding box dimensions
        analysis['width'] = analysis['x_range'][1] - analysis['x_range'][0]
        analysis['height'] = analysis['y_range'][1] - analysis['y_range'][0]
        analysis['depth'] = analysis['z_range'][1] - analysis['z_range'][0]
        
        return analysis
    
    @staticmethod
    def print_analysis(analysis: Dict, verbose: bool = True):
        """
        Print point cloud analysis results
        
        Args:
            analysis: Analysis dictionary from analyze_point_cloud()
            verbose: Whether to print detailed information
        """
        if verbose:
            print(f"\n📊 Point Cloud Analysis:")
            print(f"   Total points: {analysis['total_points']:,}")
            print(f"   X range: [{analysis['x_range'][0]:.6f}, {analysis['x_range'][1]:.6f}] (width: {analysis['width']:.6f})")
            print(f"   Y range: [{analysis['y_range'][0]:.6f}, {analysis['y_range'][1]:.6f}] (height: {analysis['height']:.6f})")
            print(f"   Z range: [{analysis['z_range'][0]:.6f}, {analysis['z_range'][1]:.6f}] (depth: {analysis['depth']:.6f})")
            print(f"   Center: ({analysis['x_center']:.6f}, {analysis['y_center']:.6f}, {analysis['z_center']:.6f})")
            print(f"   Median: ({analysis['x_median']:.6f}, {analysis['y_median']:.6f}, {analysis['z_median']:.6f})")
    
    @staticmethod
    def split_point_cloud(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                         split_method: str = 'center') -> Tuple[Dict, Dict, float]:
        """
        Split point cloud into left and right halves based on x-direction
        
        Args:
            x, y, z: Point cloud coordinates
            split_method: 'center' or 'median' for splitting method
            
        Returns:
            left_points: Dictionary with left half points
            right_points: Dictionary with right half points
            split_line: X-coordinate of the division line
        """
        # Determine split line
        if split_method == 'center':
            split_line = (np.min(x) + np.max(x)) / 2
        elif split_method == 'median':
            split_line = np.median(x)
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        # Split points
        left_mask = x < split_line
        right_mask = x >= split_line
        
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
        
        return left_points, right_points, split_line
    
    @staticmethod
    def print_split_info(left_points: Dict, right_points: Dict, split_line: float, 
                        split_method: str, total_points: int):
        """
        Print bilateral splitting information
        
        Args:
            left_points: Dictionary with left half points
            right_points: Dictionary with right half points
            split_line: X-coordinate of the division line
            split_method: Method used for splitting
            total_points: Total number of original points
        """
        print(f"\n🔄 Point Cloud Splitting:")
        print(f"   Split method: {split_method}")
        print(f"   Split line: x = {split_line:.6f}")
        print(f"   Left points: {len(left_points['x']):,} ({len(left_points['x'])/total_points*100:.1f}%)")
        print(f"   Right points: {len(right_points['x']):,} ({len(right_points['x'])/total_points*100:.1f}%)")
        print(f"   Total points: {total_points:,}")


class PointCloudSaver:
    """
    Point cloud saving utilities
    """
    
    @staticmethod
    def save_pcd_file(x: np.ndarray, y: np.ndarray, z: np.ndarray, filename: str, 
                     description: str = ""):
        """
        Save point cloud as PCD file
        
        Args:
            x, y, z: Point cloud coordinates
            filename: Output filename
            description: Optional description for identification
        """
        num_points = len(x)
        
        if description:
            print(f"💾 Saving {description} to: {filename}")
        else:
            print(f"💾 Saving point cloud to: {filename}")
        
        # Create PCD header
        header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""
        
        # Write PCD file
        with open(filename, 'w') as f:
            f.write(header)
            for i in range(num_points):
                f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f}\n")
        
        print(f"   ✅ Saved {num_points:,} points")


class PointCloudVisualizer:
    """
    Point cloud visualization utilities
    """
    
    @staticmethod
    def create_bilateral_visualization(original_data: Tuple, left_points: Dict, right_points: Dict,
                                     split_line: float, base_filename: str, 
                                     split_method: str = 'center') -> str:
        """
        Create comprehensive visualization of original and split point clouds
        
        Args:
            original_data: Tuple of (x, y, z) original coordinates
            left_points: Dictionary with left half points
            right_points: Dictionary with right half points
            split_line: X-coordinate of the division line
            base_filename: Base filename for saving plots
            split_method: Method used for splitting
            
        Returns:
            Filename of saved plot
        """
        x, y, z = original_data
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Original point cloud (top-left)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Original Point Cloud\n({len(x):,} points)')
        
        # 2. Split visualization (top-middle)
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(left_points['x'], left_points['y'], left_points['z'], 
                   c='blue', s=1, alpha=0.7, label=f'Left ({len(left_points["x"]):,})')
        ax2.scatter(right_points['x'], right_points['y'], right_points['z'], 
                   c='red', s=1, alpha=0.7, label=f'Right ({len(right_points["x"]):,})')
        
        # Draw split line
        y_range = [np.min(y), np.max(y)]
        z_range = [np.min(z), np.max(z)]
        ax2.plot([split_line, split_line], y_range, [z_range[0], z_range[0]], 
                'k--', linewidth=3, label=f'Split line (x={split_line:.3f})')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Bilateral Split Visualization')
        ax2.legend()
        
        # 3. Left half detail (top-right)
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        if len(left_points['x']) > 0:
            ax3.scatter(left_points['x'], left_points['y'], left_points['z'], 
                       c=left_points['z'], cmap='Blues', s=2, alpha=0.8)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.set_title(f'Left Half\n({len(left_points["x"]):,} points)')
        else:
            ax3.text(0.5, 0.5, 0.5, 'No left points', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Left Half - Empty')
        
        # 4. Right half detail (bottom-left)
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        if len(right_points['x']) > 0:
            ax4.scatter(right_points['x'], right_points['y'], right_points['z'], 
                       c=right_points['z'], cmap='Reds', s=2, alpha=0.8)
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_zlabel('Z')
            ax4.set_title(f'Right Half\n({len(right_points["x"]):,} points)')
        else:
            ax4.text(0.5, 0.5, 0.5, 'No right points', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Right Half - Empty')
        
        # 5. Statistical comparison (bottom-middle)
        ax5 = fig.add_subplot(2, 3, 5)
        
        if len(left_points['x']) > 0 and len(right_points['x']) > 0:
            # Compare Z-distributions
            ax5.hist(left_points['z'], bins=50, alpha=0.6, label='Left', color='blue', density=True)
            ax5.hist(right_points['z'], bins=50, alpha=0.6, label='Right', color='red', density=True)
            ax5.set_xlabel('Z coordinate')
            ax5.set_ylabel('Density')
            ax5.set_title('Z-Distribution Comparison')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Distribution\ncomparison\nnot available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Distribution Comparison - N/A')
        
        # 6. Summary statistics (bottom-right)
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calculate statistics
        summary_text = f"POINT CLOUD SPLITTING SUMMARY\n"
        summary_text += f"{'='*50}\n"
        summary_text += f"Original file: {base_filename}\n"
        summary_text += f"Split method: {split_method}\n"
        summary_text += f"Split line: x = {split_line:.6f}\n\n"
        
        summary_text += f"POINT COUNTS:\n"
        summary_text += f"  Total points: {len(x):,}\n"
        summary_text += f"  Left points:  {len(left_points['x']):,} ({len(left_points['x'])/len(x)*100:.1f}%)\n"
        summary_text += f"  Right points: {len(right_points['x']):,} ({len(right_points['x'])/len(x)*100:.1f}%)\n\n"
        
        if len(left_points['x']) > 0:
            summary_text += f"LEFT HALF STATS:\n"
            summary_text += f"  X range: [{np.min(left_points['x']):.3f}, {np.max(left_points['x']):.3f}]\n"
            summary_text += f"  Z range: [{np.min(left_points['z']):.3f}, {np.max(left_points['z']):.3f}]\n\n"
        
        if len(right_points['x']) > 0:
            summary_text += f"RIGHT HALF STATS:\n"
            summary_text += f"  X range: [{np.min(right_points['x']):.3f}, {np.max(right_points['x']):.3f}]\n"
            summary_text += f"  Z range: [{np.min(right_points['z']):.3f}, {np.max(right_points['z']):.3f}]\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{base_filename}_splitting_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as: {plot_filename}")
        
        plt.show()
        
        return plot_filename


# Convenience functions for backward compatibility and ease of use
def load_point_cloud(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function for loading point clouds"""
    return PointCloudLoader.load_point_cloud(filename)

def analyze_point_cloud(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
    """Convenience function for analyzing point clouds"""
    return PointCloudAnalyzer.analyze_point_cloud(x, y, z)

def split_point_cloud(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                     split_method: str = 'center') -> Tuple[Dict, Dict, float]:
    """Convenience function for splitting point clouds"""
    return PointCloudAnalyzer.split_point_cloud(x, y, z, split_method)

def save_pcd_file(x: np.ndarray, y: np.ndarray, z: np.ndarray, filename: str, 
                 description: str = ""):
    """Convenience function for saving PCD files"""
    return PointCloudSaver.save_pcd_file(x, y, z, filename, description)