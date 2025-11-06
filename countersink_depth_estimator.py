#!/usr/bin/env python3
"""
Countersink Depth Estimator
---------------------------
Estimates countersink depth from 3D point cloud data by fitting a cone
to the hole region and calculating the depth based on geometric parameters.

Approach:
1. Load and preprocess point cloud
2. Segment into surface plane and hole regions
3. Fit cone to hole region using least squares optimization
4. Estimate apex location and calculate depth relative to surface
5. Account for inner radius to get total effective depth

Usage:
    python countersink_depth_estimator.py [pcd_file] [--plot]
    
Requirements:
    pip install numpy scipy matplotlib open3d (optional)
"""

import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from typing import Tuple, Optional, Dict, Any

# Set random seed for reproducible results
np.random.seed(42)

class CountersinkDepthEstimator:
    def __init__(self, 
                 expected_csk_angle_deg: float = 100.0,
                 expected_inner_radius: float = 1.2446,
                 surface_thickness: float = 0.005,  # mm - thickness to consider as "surface"
                 min_hole_points: int = 50,  # minimum points needed for cone fitting
                 optimizer: str = 'slsqp',  # 'slsqp' or 'slsqp'
                 flip_z: bool = False,  # Flip z-coordinates (multiply by -1)
                 meters_to_mm: bool = False,  # Convert from meters to millimeters
                 noise_filter: bool = True,  # Apply noise filtering
                 noise_neighbors: int = 100,  # Minimum neighbors for noise filter
                 noise_radius: float = 0.5,  # Search radius for noise filter (mm)
                 lateral_filter: bool = True,  # Apply lateral distance filtering from hole center
                 lateral_filter_threshold: float = 3.0,  # Maximum lateral distance from hole center (mm)
                 visualize_lateral_filter: bool = False,  # Show lateral filtering visualization
                 z_filter: bool = True,  # Apply z-direction filtering for hole points
                 z_filter_threshold: float = 0.6,  # Maximum z distance from median z (mm)
                 outlier_method: str = 'percentile',  # 'percentile' or 'median_filter'
                 outlier_threshold: float = 0.02,  # For percentile: fraction to remove (0.02 = 2%), for median_filter: multiplier (e.g., 3.0)
                 ransac_sample_points: int = 10,  # Number of points to sample for RANSAC plane fitting
                 random_seed: int = 42):  # Random seed for reproducible RANSAC results
        
        self.expected_csk_angle_deg = expected_csk_angle_deg
        self.expected_csk_angle_rad = np.radians(expected_csk_angle_deg)
        self.expected_half_angle = self.expected_csk_angle_rad / 2
        self.expected_inner_radius = expected_inner_radius
        self.surface_thickness = surface_thickness
        self.min_hole_points = min_hole_points
        self.optimizer = optimizer.lower()
        self.flip_z = flip_z
        self.meters_to_mm = meters_to_mm
        self.noise_filter = noise_filter
        self.noise_neighbors = noise_neighbors
        self.noise_radius = noise_radius
        self.lateral_filter = lateral_filter
        self.lateral_filter_threshold = lateral_filter_threshold
        self.visualize_lateral_filter = visualize_lateral_filter
        self.z_filter = z_filter
        self.z_filter_threshold = z_filter_threshold
        self.outlier_method = outlier_method.lower()
        self.outlier_threshold = outlier_threshold
        self.ransac_sample_points = ransac_sample_points
        self.random_seed = random_seed
        
        # Validate optimizer choice
        if self.optimizer not in ['least_squares', 'slsqp']:
            raise ValueError("optimizer must be 'least_squares' or 'slsqp'")
        
        # Print initialization parameters with 6 decimal places
        print("🚀 CountersinkDepthEstimator initialized with parameters:")
        print(f"   expected_csk_angle_deg: {self.expected_csk_angle_deg:.6f}")
        print(f"   expected_inner_radius: {self.expected_inner_radius:.6f}")
        print(f"   surface_thickness: {self.surface_thickness:.6f}")
        print(f"   min_hole_points: {self.min_hole_points}")
        print(f"   optimizer: '{self.optimizer}'")
        print(f"   flip_z: {self.flip_z}")
        print(f"   meters_to_mm: {self.meters_to_mm}")
        print(f"   noise_filter: {self.noise_filter}")
        print(f"   noise_neighbors: {self.noise_neighbors}")
        print(f"   noise_radius: {self.noise_radius:.6f}")
        print(f"   lateral_filter: {self.lateral_filter}")
        print(f"   lateral_filter_threshold: {self.lateral_filter_threshold:.6f}")
        print(f"   visualize_lateral_filter: {self.visualize_lateral_filter}")
        print(f"   z_filter: {self.z_filter}")
        print(f"   z_filter_threshold: {self.z_filter_threshold:.6f}")
        print(f"   outlier_method: '{self.outlier_method}'")
        print(f"   outlier_threshold: {self.outlier_threshold:.6f}")
        print(f"   ransac_sample_points: {self.ransac_sample_points}")
        print(f"   random_seed: {self.random_seed}")
        print(f"   expected_half_angle (rad): {self.expected_half_angle:.6f}")
        
        # Results storage
        self.results = {}
    
    def preprocess_point_cloud(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply preprocessing transformations to point cloud data
        
        Args:
            x, y, z: Original point cloud coordinates
            
        Returns:
            Tuple of transformed (x, y, z) coordinates
        """
        # Create copies to avoid modifying original data
        x_proc = x.copy()
        y_proc = y.copy()
        z_proc = z.copy()
        
        # Apply transformations
        transformations_applied = []
        
        # 1. Convert from meters to millimeters
        if self.meters_to_mm:
            x_proc *= 1000.0
            y_proc *= 1000.0
            z_proc *= 1000.0
            transformations_applied.append("meters → mm conversion")
        
        # 2. Flip z-coordinates
        if self.flip_z:
            z_proc *= -1.0
            transformations_applied.append("z-axis flip")
        
        # Report applied transformations
        if transformations_applied:
            print(f"⚙️  Applied transformations: {', '.join(transformations_applied)}")
            print(f"   Original ranges: x=[{np.min(x):.6f}, {np.max(x):.6f}], "
                  f"y=[{np.min(y):.6f}, {np.max(y):.6f}], z=[{np.min(z):.6f}, {np.max(z):.6f}]")
            print(f"   Processed ranges: x=[{np.min(x_proc):.6f}, {np.max(x_proc):.6f}], "
                  f"y=[{np.min(y_proc):.6f}, {np.max(y_proc):.6f}], z=[{np.min(z_proc):.6f}, {np.max(z_proc):.6f}]")
        
        return x_proc, y_proc, z_proc
    
    def filter_noise(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter noise points based on local point density
        
        A point is considered noise if it has fewer than min_neighbors points 
        within the specified radius.
        
        Args:
            x, y, z: Point cloud coordinates
            
        Returns:
            Tuple of (x_filtered, y_filtered, z_filtered, noise_mask)
            where noise_mask is True for points that were kept
        """
        if not self.noise_filter:
            # Return all points as valid if noise filtering is disabled
            valid_mask = np.ones(len(x), dtype=bool)
            return x, y, z, valid_mask
        
        print(f"🔍 Applying noise filter (min_neighbors={self.noise_neighbors}, radius={self.noise_radius:.3f}mm)")
        
        # Stack coordinates for efficient distance calculations
        points = np.column_stack([x, y, z])
        n_points = len(points)
        
        # For large point clouds, use spatial indexing for efficiency
        if n_points > 10000:
            # Use sklearn KDTree for large datasets
            try:
                from sklearn.neighbors import NearestNeighbors
                
                # Build nearest neighbors index
                nbrs = NearestNeighbors(n_neighbors=min(self.noise_neighbors + 1, n_points), 
                                      radius=self.noise_radius, 
                                      algorithm='kd_tree')
                nbrs.fit(points)
                
                # Find neighbors within radius for each point
                distances, indices = nbrs.radius_neighbors(points, radius=self.noise_radius)
                
                # Count neighbors (excluding the point itself)
                neighbor_counts = np.array([len(neighbors) - 1 for neighbors in indices])
                
            except ImportError:
                print("⚠️  sklearn not available, using brute force method (may be slow)")
                neighbor_counts = self._count_neighbors_brute_force(points, self.noise_radius)
        else:
            # Use brute force for smaller datasets
            neighbor_counts = self._count_neighbors_brute_force(points, self.noise_radius)
        
        # Create mask for points with sufficient neighbors
        valid_mask = neighbor_counts >= self.noise_neighbors
        
        # Apply filter
        x_filtered = x[valid_mask]
        y_filtered = y[valid_mask]
        z_filtered = z[valid_mask]
        
        # Report results
        noise_points = n_points - np.sum(valid_mask)
        noise_percentage = (noise_points / n_points) * 100
        
        print(f"   📊 Noise filtering results:")
        print(f"      Original points: {n_points:,}")
        print(f"      Noise points removed: {noise_points:,} ({noise_percentage:.1f}%)")
        print(f"      Valid points retained: {len(x_filtered):,} ({100-noise_percentage:.1f}%)")
        
        return x_filtered, y_filtered, z_filtered, valid_mask
    
    def _count_neighbors_brute_force(self, points: np.ndarray, radius: float) -> np.ndarray:
        """Count neighbors using brute force method"""
        n_points = len(points)
        neighbor_counts = np.zeros(n_points, dtype=int)
        
        # Calculate pairwise distances and count neighbors
        for i in range(n_points):
            distances = np.linalg.norm(points - points[i], axis=1)
            # Count neighbors within radius (excluding the point itself)
            neighbor_counts[i] = np.sum((distances <= radius) & (distances > 0))
        
        return neighbor_counts
    
    def _filter_lateral_outliers(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter outliers based on lateral distance from hole center
        
        This method identifies points that are too far from the expected countersink hole center
        in the x-y plane. It estimates the hole center as the centroid of all points and removes
        points that are more than lateral_filter_threshold away from this center.
        
        Args:
            x, y, z: Point cloud coordinates (already filtered by density-based method)
            
        Returns:
            Tuple of (x_filtered, y_filtered, z_filtered, lateral_mask)
            where lateral_mask is True for points that were kept
        """
        if not self.lateral_filter:
            # Return all points as valid if lateral filtering is disabled
            valid_mask = np.ones(len(x), dtype=bool)
            return x, y, z, valid_mask
        
        print(f"🔍 Applying lateral distance filter (threshold={self.lateral_filter_threshold:.3f}mm)")
        
        n_points = len(x)
        if n_points == 0:
            return x, y, z, np.array([], dtype=bool)
        
        # Estimate hole center as median of all points (x median and y median separately)
        center_x = np.median(x)
        center_y = np.median(y)
        
        # Calculate lateral distances from center
        lateral_distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create mask for points within threshold
        valid_mask = lateral_distances <= self.lateral_filter_threshold
        
        # Apply filter
        x_filtered = x[valid_mask]
        y_filtered = y[valid_mask]
        z_filtered = z[valid_mask]
        
        # Report results
        outlier_points = n_points - np.sum(valid_mask)
        outlier_percentage = (outlier_points / n_points) * 100
        
        print(f"   📊 Lateral filtering results:")
        print(f"      Hole center estimate: ({center_x:.3f}, {center_y:.3f}) mm")
        print(f"      Max lateral distance: {np.max(lateral_distances):.3f} mm")
        print(f"      Mean lateral distance: {np.mean(lateral_distances):.3f} mm")
        print(f"      Lateral outliers removed: {outlier_points:,} ({outlier_percentage:.1f}%)")
        print(f"      Points retained: {len(x_filtered):,} ({100-outlier_percentage:.1f}%)")
        
        return x_filtered, y_filtered, z_filtered, valid_mask

    def _filter_hole_lateral_outliers(self, hole_points: Dict) -> Tuple[Dict, np.ndarray]:
        """
        Filter lateral outliers from hole points based on distance from hole center
        
        This method applies lateral filtering specifically to hole points after surface/hole
        segmentation. It estimates the hole center and removes hole points that are too far
        from this center in the x-y plane.
        
        Args:
            hole_points: Dictionary with 'x', 'y', 'z' arrays for hole points
            
        Returns:
            Tuple of (filtered_hole_points_dict, lateral_mask)
            where lateral_mask is True for points that were kept
        """
        x, y, z = hole_points['x'], hole_points['y'], hole_points['z']
        n_points = len(x)
        
        if n_points == 0:
            return hole_points, np.array([], dtype=bool)
        
        print(f"🔍 Applying lateral distance filter to {n_points} hole points (threshold={self.lateral_filter_threshold:.3f}mm)")
        
        # Estimate hole center as median of hole points (x median and y median separately)
        center_x = np.median(x)
        center_y = np.median(y)
        
        # Calculate lateral distances from hole center
        lateral_distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create mask for points within threshold
        valid_mask = lateral_distances <= self.lateral_filter_threshold
        
        # Apply filter
        filtered_hole_points = {
            'x': x[valid_mask],
            'y': y[valid_mask],
            'z': z[valid_mask]
        }
        
        # Report results
        outlier_points = n_points - np.sum(valid_mask)
        outlier_percentage = (outlier_points / n_points) * 100
        
        print(f"   📊 Hole lateral filtering results:")
        print(f"      Hole center estimate: ({center_x:.3f}, {center_y:.3f}) mm")
        print(f"      Max lateral distance: {np.max(lateral_distances):.3f} mm")
        print(f"      Mean lateral distance: {np.mean(lateral_distances):.3f} mm")
        print(f"      Hole outliers removed: {outlier_points:,} ({outlier_percentage:.1f}%)")
        print(f"      Hole points retained: {len(filtered_hole_points['x']):,} ({100-outlier_percentage:.1f}%)")
        
        # Show visualization if enabled
        if self.visualize_lateral_filter:
            self.visualize_lateral_filtering(hole_points, show_plot=True, save_path=None)
        
        return filtered_hole_points, valid_mask

    def visualize_lateral_filtering(self, hole_points: Dict, show_plot: bool = True, save_path: str = None) -> None:
        """
        Visualize the lateral filtering process for hole points
        
        Creates a 2D plot showing:
        - Original hole points
        - Filtered hole points (kept)
        - Removed outliers
        - Hole center estimate
        - Threshold circle
        
        Args:
            hole_points: Dictionary with 'x', 'y', 'z' arrays for hole points
            show_plot: Whether to display the plot
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        x, y, z = hole_points['x'], hole_points['y'], hole_points['z']
        n_points = len(x)
        
        if n_points == 0:
            print("No hole points to visualize")
            return
        
        # Calculate filtering results (using median center for consistency)
        center_x = np.median(x)
        center_y = np.median(y)
        lateral_distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        valid_mask = lateral_distances <= self.lateral_filter_threshold
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Before filtering
        ax1.scatter(x, y, c=z, cmap='viridis', alpha=0.7, s=20)
        ax1.scatter(center_x, center_y, c='red', marker='x', s=100, linewidth=3, label='Hole Center')
        
        # Add threshold circle
        circle1 = patches.Circle((center_x, center_y), self.lateral_filter_threshold, 
                                fill=False, color='red', linestyle='--', linewidth=2, 
                                label=f'Threshold ({self.lateral_filter_threshold:.1f}mm)')
        ax1.add_patch(circle1)
        
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title(f'Before Lateral Filtering\n{n_points} hole points')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: After filtering
        kept_points = valid_mask
        removed_points = ~valid_mask
        
        # Plot kept points
        if np.any(kept_points):
            scatter1 = ax2.scatter(x[kept_points], y[kept_points], c=z[kept_points], 
                                 cmap='viridis', alpha=0.7, s=20, label='Kept Points')
        
        # Plot removed points
        if np.any(removed_points):
            ax2.scatter(x[removed_points], y[removed_points], c='red', 
                       marker='x', alpha=0.8, s=30, label='Removed Outliers')
        
        # Add hole center and threshold circle
        ax2.scatter(center_x, center_y, c='red', marker='x', s=100, linewidth=3, label='Hole Center')
        circle2 = patches.Circle((center_x, center_y), self.lateral_filter_threshold, 
                                fill=False, color='red', linestyle='--', linewidth=2, 
                                label=f'Threshold ({self.lateral_filter_threshold:.1f}mm)')
        ax2.add_patch(circle2)
        
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        n_kept = np.sum(kept_points)
        n_removed = np.sum(removed_points)
        ax2.set_title(f'After Lateral Filtering\n{n_kept} kept, {n_removed} removed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Make sure both plots have the same scale
        all_x = x
        all_y = y
        margin = max(self.lateral_filter_threshold * 0.2, 0.5)
        x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
        y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin
        
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        
        # Add colorbar for depth information
        if np.any(kept_points):
            cbar = plt.colorbar(scatter1, ax=ax2, shrink=0.8)
            cbar.set_label('Depth (mm)')
        
        # Add statistics text
        stats_text = f"""Statistics:
Original points: {n_points}
Kept points: {n_kept} ({n_kept/n_points*100:.1f}%)
Removed points: {n_removed} ({n_removed/n_points*100:.1f}%)
Max distance: {np.max(lateral_distances):.2f} mm
Mean distance: {np.mean(lateral_distances):.2f} mm
Threshold: {self.lateral_filter_threshold:.2f} mm"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Lateral filtering visualization saved to: {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def _filter_hole_z_outliers(self, hole_points: Dict) -> Tuple[Dict, np.ndarray]:
        """
        Filter z-direction outliers from hole points based on distance from median z
        
        This method applies z-direction filtering specifically to hole points after surface/hole
        segmentation. It calculates the median z value and removes hole points that are too far
        from this median in the z direction.
        
        Args:
            hole_points: Dictionary with 'x', 'y', 'z' arrays for hole points
            
        Returns:
            Tuple of (filtered_hole_points_dict, z_mask)
            where z_mask is True for points that were kept
        """
        x, y, z = hole_points['x'], hole_points['y'], hole_points['z']
        n_points = len(x)
        
        if n_points == 0:
            return hole_points, np.array([], dtype=bool)
        
        print(f"🔍 Applying z-direction filter to {n_points} hole points (threshold={self.z_filter_threshold:.3f}mm)")
        
        # Calculate median z value
        median_z = np.median(z)
        
        # Calculate z distances from median
        z_distances = np.abs(z - median_z)
        
        # Create mask for points within threshold
        valid_mask = z_distances <= self.z_filter_threshold
        
        # Apply filter
        filtered_hole_points = {
            'x': x[valid_mask],
            'y': y[valid_mask],
            'z': z[valid_mask]
        }
        
        # Report results
        outlier_points = n_points - np.sum(valid_mask)
        outlier_percentage = (outlier_points / n_points) * 100
        
        print(f"   📊 Hole z-direction filtering results:")
        print(f"      Median z value: {median_z:.3f} mm")
        print(f"      Max z distance from median: {np.max(z_distances):.3f} mm")
        print(f"      Mean z distance from median: {np.mean(z_distances):.3f} mm")
        print(f"      Hole z outliers removed: {outlier_points:,} ({outlier_percentage:.1f}%)")
        print(f"      Hole points retained: {len(filtered_hole_points['x']):,} ({100-outlier_percentage:.1f}%)")
        
        return filtered_hole_points, valid_mask

    def load_point_cloud(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load point cloud from PCD or CSV file"""
        if filename.lower().endswith('.pcd'):
            return self._load_pcd(filename)
        elif filename.lower().endswith('.csv'):
            return self._load_csv(filename)
        else:
            raise ValueError("Unsupported file format. Use .pcd or .csv")
    
    def _load_pcd(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load PCD file"""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(filename)
            points = np.asarray(pcd.points)
            return points[:, 0], points[:, 1], points[:, 2]
        except ImportError:
            # Fallback: parse PCD manually
            return self._parse_pcd_manual(filename)
    
    def _parse_pcd_manual(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Manual PCD parser (fallback when Open3D not available)"""
        points = []
        with open(filename, 'r') as f:
            data_section = False
            for line in f:
                if line.startswith('DATA ascii'):
                    data_section = True
                    continue
                if data_section and len(line.strip()) > 0:
                    try:
                        x, y, z = map(float, line.strip().split())
                        points.append([x, y, z])
                    except:
                        continue
        
        points = np.array(points)
        return points[:, 0], points[:, 1], points[:, 2]
    
    def _load_csv(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load CSV file"""
        import csv
        points = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x, y, z = float(row['x']), float(row['y']), float(row['z'])
                    points.append([x, y, z])
                except:
                    continue
        
        points = np.array(points)
        return points[:, 0], points[:, 1], points[:, 2]
    
    def estimate_surface_plane(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate the dominant surface plane using RANSAC-like approach
        Returns: (surface_normal, surface_point, surface_distance_threshold)
        """
        points = np.column_stack([x, y, z])
        
        # Method 1: Try PCA on points with highest z-values (likely surface points)
        # Take top 50% of points by z-value as initial surface candidates
        z_percentile = np.percentile(z, 50)
        surface_candidates = points[z >= z_percentile]
        
        if len(surface_candidates) < 10:
            # Fallback: use all points
            surface_candidates = points
        
        # Use PCA to find the dominant plane
        centroid = np.mean(surface_candidates, axis=0)
        centered_points = surface_candidates - centroid
        
        # Compute covariance matrix and find principal components
        cov_matrix = np.cov(centered_points.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # The normal is the eigenvector with the smallest eigenvalue
        # (direction of least variation = normal to plane)
        normal_idx = np.argmin(eigenvals)
        surface_normal = eigenvecs[:, normal_idx]
        
        # Ensure normal points "upward" (positive component in dominant direction)
        if np.sum(surface_normal) < 0:
            surface_normal = -surface_normal
            
        # Refine using RANSAC-like approach with configurable sample points
        best_normal, best_point = self._ransac_plane_fit(surface_candidates, surface_normal, centroid, 
                                                        n_sample_points=self.ransac_sample_points)
        
        # Determine thickness threshold based on point cloud density and surface quality
        distances_to_plane = np.abs(np.dot(points - best_point, best_normal))
        distance_std = np.std(distances_to_plane)
        
        # Use a more conservative threshold for better surface/hole separation
        # Take the smaller of: user-specified thickness or 1.5x standard deviation
        thickness_threshold = 0.8 * distance_std #min(self.surface_thickness, 1.5 * distance_std)
        
        # Ensure minimum threshold for noisy data
        # thickness_threshold = max(thickness_threshold, 0.1)  # minimum 0.1mm threshold
        
        print(f"Estimated surface normal: ({best_normal[0]:.3f}, {best_normal[1]:.3f}, {best_normal[2]:.3f})")
        print(f"Surface point: ({best_point[0]:.3f}, {best_point[1]:.3f}, {best_point[2]:.3f})")
        print(f"Surface thickness threshold: {thickness_threshold:.3f} mm")
        
        return best_normal, best_point, thickness_threshold
    
    def _ransac_plane_fit(self, points: np.ndarray, initial_normal: np.ndarray, initial_point: np.ndarray, 
                         n_iterations: int = 100, n_sample_points: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RANSAC-based plane fitting for robust surface estimation
        
        Args:
            points: Surface candidate points
            initial_normal: Initial estimate of surface normal
            initial_point: Initial estimate of surface point
            n_iterations: Number of RANSAC iterations
            n_sample_points: Number of points to sample for each plane fit (default: 10)
            
        Returns:
            Tuple of (best_normal, best_point, inlier_mask)
            where inlier_mask is a boolean array indicating which points are inliers
        """
        # Reset random seed for deterministic RANSAC results
        np.random.seed(self.random_seed)
        
        best_normal = initial_normal.copy()
        best_point = initial_point.copy()
        best_inlier_count = 0
        best_inlier_mask = np.zeros(len(points), dtype=bool)
        
        # Ensure we don't sample more points than available
        n_sample_points = min(n_sample_points, len(points))
        
        for iteration in range(n_iterations):
            # Sample more points for better plane estimation
            if len(points) < max(3, n_sample_points):
                break
                
            sample_indices = np.random.choice(len(points), n_sample_points, replace=False)
            sample_points = points[sample_indices]
            
            # Fit plane using least squares on the sampled points
            # Center the points
            centroid = np.mean(sample_points, axis=0)
            centered_points = sample_points - centroid
            
            # Use SVD for robust plane fitting
            try:
                # Compute covariance matrix
                cov_matrix = np.cov(centered_points.T)
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                
                # Normal is eigenvector with smallest eigenvalue
                normal_idx = np.argmin(eigenvals)
                normal = eigenvecs[:, normal_idx]
                
                if np.linalg.norm(normal) < 1e-6:
                    continue  # Degenerate case
                    
                normal = normal / np.linalg.norm(normal)
                
                # Ensure consistent orientation with initial normal
                if np.dot(normal, initial_normal) < 0:
                    normal = -normal
                
                plane_point = centroid
                
                # Count inliers using all points (not just sampled ones)
                distances = np.abs(np.dot(points - plane_point, normal))
                current_inlier_mask = distances < self.surface_thickness
                inliers = np.sum(current_inlier_mask)
                
                if inliers > best_inlier_count:
                    best_inlier_count = inliers
                    best_normal = normal
                    best_point = plane_point
                    best_inlier_mask = current_inlier_mask
                    
            except np.linalg.LinAlgError:
                # Skip this iteration if SVD fails
                continue
        
        # Calculate final statistics
        final_distances = np.abs(np.dot(points - best_point, best_normal))
        inlier_std = np.std(final_distances[best_inlier_mask]) if np.sum(best_inlier_mask) > 0 else 0.0
        
        print(f"  RANSAC plane fit: {best_inlier_count} inliers from {len(points)} candidates using {n_sample_points} sample points")
        print(f"  Inlier distance std dev: {inlier_std:.4f} mm")
        
        return best_normal, best_point, best_inlier_mask

    def segment_surface_and_hole(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Segment point cloud into surface and hole regions based on surface normal estimation
        Returns dictionaries with 'x', 'y', 'z' arrays for each region
        """
        points = np.column_stack([x, y, z])
        
        # Estimate the dominant surface plane
        surface_normal, surface_point, thickness_threshold = self.estimate_surface_plane(x, y, z)
        
        # Calculate signed distances from each point to the surface plane
        # Positive distance = above surface, negative = below surface
        distances_to_surface = np.dot(points - surface_point, surface_normal)
        
        # Classify points based on distance to surface plane
        surface_mask = np.abs(distances_to_surface) <= thickness_threshold
        hole_mask = distances_to_surface < -thickness_threshold
        
        surface_points = {
            'x': x[surface_mask],
            'y': y[surface_mask], 
            'z': z[surface_mask]
        }
        
        hole_points = {
            'x': x[hole_mask],
            'y': y[hole_mask],
            'z': z[hole_mask]
        }
        
        print(f"Surface points: {len(surface_points['x'])}")
        print(f"Hole points: {len(hole_points['x'])}")
        if len(hole_points['x']) > 0:
            print(f"Hole distance range: {distances_to_surface[hole_mask].min():.3f} to {distances_to_surface[hole_mask].max():.3f} mm from surface")
            print(f"Hole xy range: x=[{x[hole_mask].min():.3f}, {x[hole_mask].max():.3f}], y=[{y[hole_mask].min():.3f}, {y[hole_mask].max():.3f}]")
        
        # Apply lateral filtering to hole points only (if enabled)
        if self.lateral_filter and len(hole_points['x']) > 0:
            hole_points_filtered, hole_lateral_mask = self._filter_hole_lateral_outliers(hole_points)
            hole_points = hole_points_filtered
            
            print(f"Hole points after lateral filtering: {len(hole_points['x'])}")
        
        # Apply z-direction filtering to hole points only (if enabled)
        if self.z_filter and len(hole_points['x']) > 0:
            hole_points_filtered, hole_z_mask = self._filter_hole_z_outliers(hole_points)
            hole_points = hole_points_filtered
            
            print(f"Hole points after z-direction filtering: {len(hole_points['x'])}")
        
        # Store surface plane information for later use
        self.results['surface_normal'] = surface_normal
        self.results['surface_point'] = surface_point  
        self.results['surface_thickness'] = thickness_threshold
        self.results['n_surface_points'] = len(surface_points['x'])
        self.results['n_hole_points'] = len(hole_points['x'])
        
        return surface_points, hole_points
    
    def fit_cone_to_points(self, hole_points: Dict) -> Dict:
        """
        Fit cone to hole points using selected optimization method with fallback.
        
        Cone parameterization:
        - Apex: (ax, ay, az)
        - Axis direction: (dx, dy, dz) - normalized
        - Half-angle: theta (fixed)
        
        For a point (x,y,z), distance to cone surface is:
        d = ||(P-A) - ((P-A)·D)D|| - ||(P-A)·D|| * tan(theta)
        """
        
        # Try primary method first
        primary_method = self.optimizer
        if primary_method == 'slsqp':
            try:
                print(f"  Trying primary method: SLSQP-AA")
                return self._fit_cone_slsqp_AA(hole_points)
            except Exception as e:
                print(f"  Primary SLSQP-AA failed: {e}")
                print(f"  Falling back to least_squares method...")
                try:
                    return self._fit_cone_least_squares(hole_points)
                except Exception as e2:
                    print(f"  Fallback least_squares also failed: {e2}")
                    print(f"  Trying original SLSQP method...")
                    return self._fit_cone_slsqp(hole_points)
        else:
            try:
                print(f"  Trying primary method: least_squares")
                return self._fit_cone_least_squares(hole_points)
            except Exception as e:
                print(f"  Primary least_squares failed: {e}")
                print(f"  Falling back to SLSQP-AA method...")
                try:
                    return self._fit_cone_slsqp_AA(hole_points)
                except Exception as e2:
                    print(f"  Fallback SLSQP-AA also failed: {e2}")
                    print(f"  Trying original SLSQP method...")
                    return self._fit_cone_slsqp(hole_points)
    
    def _fit_cone_slsqp_AA(self, hole_points: Dict) -> Dict:
        """
        Alternative cone fitting using SLSQP optimizer with adaptive outlier removal (AA).
        This method removes the top 10% highest cost points during optimization.
        """
        
        x, y, z = hole_points['x'], hole_points['y'], hole_points['z']
        
        if len(x) < self.min_hole_points:
            raise ValueError(f"Insufficient points for cone fitting: {len(x)} < {self.min_hole_points}")
        
        # Initial guess (same as other methods)
        hole_centroid = [np.median(x), np.median(y), np.median(z)]
        hole_radius_estimate = np.sqrt(np.median(x**2 + y**2))
        
        if hole_radius_estimate > 0:
            estimated_apex_offset = hole_radius_estimate / np.tan(self.expected_half_angle)
            initial_apex = [hole_centroid[0], hole_centroid[1], 
                           hole_centroid[2] - estimated_apex_offset]
        else:
            initial_apex = [hole_centroid[0], hole_centroid[1], z.min() - 0.5]

        initial_apex = [hole_centroid[0], hole_centroid[1], self.results['surface_point'][2] - 1.5] 
        initial_axis = [0, 0, 1]
        
        print(f"  Initial apex estimate (SLSQP-AA): ({initial_apex[0]:.3f}, {initial_apex[1]:.3f}, {initial_apex[2]:.3f})")
        print(f"  Estimated hole radius: {hole_radius_estimate:.3f} mm")
        
        # Parameters: [ax, ay, az, dx, dy, dz]
        initial_params = np.array(initial_apex + initial_axis)
        fixed_theta = self.expected_half_angle
        
        # Outlier removal parameters - now configurable
        outlier_method = self.outlier_method  # 'percentile' or 'median_filter'
        outlier_threshold = self.outlier_threshold  # fraction for percentile, multiplier for median_filter
        
        def objective_function_with_adaptive_outlier_removal(params):
            """
            Objective function with configurable outlier removal.
            Supports two methods:
            1. 'percentile': Remove top X% highest cost points (original method)
            2. 'median_filter': Remove costs that are X times larger than median cost
            Returns total cost after removing outliers.
            """
            ax, ay, az, dx, dy, dz = params
            
            # Normalize axis direction
            axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
            if axis_norm < 1e-6:
                return 1e6  # Large penalty for degenerate case
            
            dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
            
            # Calculate residuals for ALL points
            px, py, pz = x - ax, y - ay, z - az
            proj_length = px*dx + py*dy + pz*dz
            
            # Only consider points projecting positively along axis
            valid_mask = proj_length > 0
            n_valid = np.sum(valid_mask)
            
            if n_valid < self.min_hole_points:
                return 1e6  # Penalty if too few valid projections
            
            # Calculate individual point costs
            perp_x = px - proj_length*dx
            perp_y = py - proj_length*dy
            perp_z = pz - proj_length*dz
            perp_dist = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
            
            # Expected radius at this height
            expected_radius = proj_length * np.tan(fixed_theta)
            residuals = perp_dist - expected_radius
            
            # Calculate individual costs (squared residuals for valid points)
            point_costs = np.full(len(x), 1e6)  # Initialize with high cost for invalid points
            point_costs[valid_mask] = residuals[valid_mask]**2
            
            # Apply additional penalty for invalid projections
            point_costs[~valid_mask] = 1e6
            
            # Apply outlier removal based on selected method
            valid_costs = point_costs[valid_mask]
            
            if outlier_method == 'median_filter':
                # Method 2: Median filter - remove costs that are X times larger than median
                median_cost = np.median(valid_costs)
                cost_threshold = median_cost * outlier_threshold
                keep_mask = point_costs <= cost_threshold
                
                # print(f"    Median filter: median_cost={median_cost:.6f}, threshold={cost_threshold:.6f} (x{outlier_threshold})")
                
            else:  # Default to 'percentile' method
                # Method 1: Percentile-based removal (original method)
                outlier_percentage = outlier_threshold  # Use threshold as percentage
                cost_threshold = np.percentile(valid_costs, (1.0 - outlier_percentage) * 100)
                keep_mask = point_costs <= cost_threshold
                
                # print(f"    Percentile filter: removing top {outlier_percentage*100:.1f}%, threshold={cost_threshold:.6f}")
            
            # Ensure we have enough points
            n_kept = np.sum(keep_mask)
            if n_kept < self.min_hole_points:
                # If too aggressive, keep more points with a backup strategy
                if outlier_method == 'median_filter':
                    # For median filter, increase the multiplier
                    backup_threshold = median_cost * (outlier_threshold * 2.0)  # Double the multiplier
                    keep_mask = point_costs <= backup_threshold
                else:
                    # For percentile, keep top 90%
                    cost_threshold = np.percentile(valid_costs, 90)
                    keep_mask = point_costs <= cost_threshold
                
                n_kept = np.sum(keep_mask)
                print(f"    Backup strategy applied: now keeping {n_kept} points")
                
                if n_kept < self.min_hole_points:
                    return 1e6  # Still not enough points
            
            # Calculate total cost using kept points only
            kept_costs = point_costs[keep_mask]
            total_cost = np.sum(kept_costs)
            
            return total_cost
        
        def calculate_detailed_costs(params):
            """
            Calculate detailed cost information for each point (for reporting).
            Returns individual costs and outlier identification.
            """
            ax, ay, az, dx, dy, dz = params
            
            # Normalize axis direction
            axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
            if axis_norm < 1e-6:
                return None, None, None
            
            dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
            
            # Calculate residuals for ALL points
            px, py, pz = x - ax, y - ay, z - az
            proj_length = px*dx + py*dy + pz*dz
            
            # Only consider points projecting positively along axis
            valid_mask = proj_length > 0
            n_valid = np.sum(valid_mask)
            
            if n_valid < self.min_hole_points:
                return None, None, None
            
            # Calculate individual point costs
            perp_x = px - proj_length*dx
            perp_y = py - proj_length*dy
            perp_z = pz - proj_length*dz
            perp_dist = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
            
            # Expected radius at this height
            expected_radius = proj_length * np.tan(fixed_theta)
            residuals = perp_dist - expected_radius
            
            # Calculate individual costs (squared residuals for valid points)
            point_costs = np.full(len(x), 1e6)  # Initialize with high cost for invalid points
            point_costs[valid_mask] = residuals[valid_mask]**2
            
            # Determine outliers based on configured method
            valid_costs = point_costs[valid_mask]
            
            if outlier_method == 'median_filter':
                # Use median filter method for reporting
                median_cost = np.median(valid_costs)
                cost_threshold = median_cost * outlier_threshold
            else:
                # Use percentile method for reporting
                cost_threshold = np.percentile(valid_costs, (1.0 - outlier_threshold) * 100)
            
            # Create outlier mask
            outlier_mask = (point_costs > cost_threshold) | (~valid_mask)
            inlier_mask = ~outlier_mask
            
            # Ensure we have enough inliers
            n_inliers = np.sum(inlier_mask)
            if n_inliers < self.min_hole_points:
                # If too aggressive, keep more points
                cost_threshold = np.percentile(valid_costs, 90)  # Keep top 90%
                outlier_mask = (point_costs > cost_threshold) | (~valid_mask)
                inlier_mask = ~outlier_mask
            
            return point_costs, inlier_mask, outlier_mask
        
        # Bounds for parameters [ax, ay, az, dx, dy, dz]
        bounds = [
            (x.min()-2, x.max()+2),  # ax
            (y.min()-2, y.max()+2),  # ay
            (z.min()-5, z.max()+2),  # az
            (-1, 1),    # dx
            (-1, 1),    # dy
            (-1, 1),    # dz
        ]
        
        # Constraint to keep axis normalized
        def axis_constraint(params):
            """Constraint to ensure axis is normalized"""
            dx, dy, dz = params[3], params[4], params[5]
            return dx**2 + dy**2 + dz**2 - 1.0
        
        constraints = {'type': 'eq', 'fun': axis_constraint}
        
        try:
            print(f"  Starting SLSQP-AA optimization with {len(x)} points...")
            print(f"  Outlier removal method: {outlier_method}, threshold: {outlier_threshold}")
            
            # Run optimization with adaptive outlier removal
            result = minimize(objective_function_with_adaptive_outlier_removal, 
                            initial_params,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': True})
            
            if not result.success:
                raise RuntimeError(f"SLSQP-AA optimization failed: {result.message}")
            
            # Extract final parameters
            ax, ay, az, dx, dy, dz = result.x
            
            # Ensure axis is normalized
            axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
            dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
            
            # Calculate detailed costs and outlier identification for final parameters
            point_costs, final_inlier_mask, final_outlier_mask = calculate_detailed_costs(result.x)
            
            if point_costs is None:
                raise RuntimeError("Failed to calculate final outlier identification")
            
            n_outliers_final = np.sum(final_outlier_mask)
            n_inliers_final = np.sum(final_inlier_mask)
            
            # Calculate final RMSE using all original points (for comparison)
            px, py, pz = x - ax, y - ay, z - az
            proj_length = px*dx + py*dy + pz*dz
            valid_mask = proj_length > 0
            
            perp_x = px - proj_length*dx
            perp_y = py - proj_length*dy
            perp_z = pz - proj_length*dz
            perp_dist = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
            
            expected_radius = proj_length * np.tan(fixed_theta)
            final_residuals = perp_dist - expected_radius
            final_residuals[~valid_mask] = np.abs(final_residuals[~valid_mask]) * 10
            
            rmse = np.sqrt(np.mean(final_residuals**2))
            
            # Store outlier information for visualization (ONLY from hole points used for fitting)
            outlier_coords = {
                'x': x[final_outlier_mask],
                'y': y[final_outlier_mask], 
                'z': z[final_outlier_mask]
            }
            inlier_coords = {
                'x': x[final_inlier_mask],
                'y': y[final_inlier_mask],
                'z': z[final_inlier_mask]
            }
            
            # Verification: check that we're only using hole points
            total_outlier_inlier = len(outlier_coords['x']) + len(inlier_coords['x'])
            print(f"  🔍 SLSQP-AA Outlier/Inlier verification:")
            print(f"      Original hole points: {len(x)}")
            print(f"      Inliers: {len(inlier_coords['x'])}")
            print(f"      Outliers: {len(outlier_coords['x'])}")
            print(f"      Total (inliers + outliers): {total_outlier_inlier}")
            print(f"      ✅ Consistency check: {total_outlier_inlier == len(x)}")
            
            fit_results = {
                'apex': np.array([ax, ay, az]),
                'axis': np.array([dx, dy, dz]),
                'half_angle_rad': fixed_theta,
                'half_angle_deg': np.degrees(fixed_theta),
                'included_angle_deg': np.degrees(2*fixed_theta),
                'rmse': rmse,
                'success': result.success,
                'fixed_angle': True,
                'n_outliers_removed': n_outliers_final,
                'n_inliers_used': n_inliers_final,
                'outlier_removal_applied': n_outliers_final > 0,
                'outlier_coords': outlier_coords,
                'inlier_coords': inlier_coords,
                'original_coords': {'x': x, 'y': y, 'z': z},
                'optimizer': 'SLSQP-AA',
                'optimization_cost': result.fun,
                'n_iterations': result.nit,
                'outlier_method': outlier_method,
                'outlier_threshold': outlier_threshold
            }
            
            print(f"Cone fitting results (SLSQP-AA):")
            print(f"  Apex: ({ax:.3f}, {ay:.3f}, {az:.3f})")
            print(f"  Axis: ({dx:.3f}, {dy:.3f}, {dz:.3f})")
            print(f"  Half-angle: {np.degrees(fixed_theta):.1f}° (included: {np.degrees(2*fixed_theta):.1f}°) [FIXED]")
            print(f"  RMSE: {rmse:.4f} mm")
            print(f"  Optimization cost: {result.fun:.6f}")
            print(f"  Iterations: {result.nit}")
            print(f"  Outlier removal: {n_outliers_final} outliers removed from {len(x)} points using {outlier_method} method")
            
            return fit_results
            
        except Exception as e:
            print(f"Cone fitting (SLSQP-AA) failed: {e}")
            raise
    
    def _fit_cone_least_squares(self, hole_points: Dict) -> Dict:
        """
        Original cone fitting using least squares optimization with outlier removal
        """
        
        x, y, z = hole_points['x'], hole_points['y'], hole_points['z']
        
        if len(x) < self.min_hole_points:
            raise ValueError(f"Insufficient points for cone fitting: {len(x)} < {self.min_hole_points}")
        
        # Improved initial guess based on cone geometry
        # Find the centroid of all hole points for better initial estimate
        hole_centroid = [np.mean(x), np.mean(y), np.mean(z)]
        
        # Estimate initial apex position using expected geometry
        # The apex should be approximately at the extrapolated tip of the cone
        hole_radius_estimate = np.sqrt(np.mean(x**2 + y**2))  # RMS radius
        
        # Using cone geometry: if we're at radius R and half-angle θ, 
        # the apex is at distance R/tan(θ) further along the axis
        if hole_radius_estimate > 0:
            estimated_apex_offset = hole_radius_estimate / np.tan(self.expected_half_angle)
            # Assume surface normal points "up", so apex is deeper (more negative z)
            initial_apex = [hole_centroid[0], hole_centroid[1], 
                           hole_centroid[2] - estimated_apex_offset]
        else:
            # Fallback for very small holes
            initial_apex = [hole_centroid[0], hole_centroid[1], z.min() - 0.5]
        
        # Initialize axis pointing toward surface (opposite to expected surface normal direction)
        # For typical scanning setup, this is usually +Z direction
        initial_axis = [0, 0, 1]
        
        print(f"  Initial apex estimate: ({initial_apex[0]:.3f}, {initial_apex[1]:.3f}, {initial_apex[2]:.3f})")
        print(f"  Estimated hole radius: {hole_radius_estimate:.3f} mm")
        
        # Parameters: [ax, ay, az, dx, dy, dz] - angle is fixed at expected value
        initial_params = initial_apex + initial_axis
        
        # Use fixed angle and inner radius
        fixed_theta = self.expected_half_angle
        
        def cone_residuals(params):
            ax, ay, az, dx, dy, dz = params
            
            # Normalize axis direction
            axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
            if axis_norm < 1e-6:
                return np.full(len(x), 1e6)  # Large residual for degenerate case
            
            dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
            
            # Use fixed angle
            theta = fixed_theta
            
            # Vector from apex to each point
            px, py, pz = x - ax, y - ay, z - az
            
            # Project onto axis
            proj_length = px*dx + py*dy + pz*dz
            
            # Only consider points that project in the positive direction along axis
            # (i.e., points that are on the "outside" of the cone from the apex)
            valid_mask = proj_length > 0
            
            # Perpendicular distance from axis
            perp_x = px - proj_length*dx
            perp_y = py - proj_length*dy  
            perp_z = pz - proj_length*dz
            perp_dist = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
            
            # Expected radius at this height for cone
            expected_radius = proj_length * np.tan(theta)
            
            # Residual is difference between actual and expected radius
            residuals = perp_dist - expected_radius
            
            # Penalize points that are on the wrong side of the apex
            residuals[~valid_mask] = np.abs(residuals[~valid_mask]) * 10
            
            return residuals
        
        # Bounds for parameters [ax, ay, az, dx, dy, dz] - no theta parameter
        bounds = [
            (x.min()-2, x.max()+2),  # ax - within point cloud bounds
            (y.min()-2, y.max()+2),  # ay - within point cloud bounds
            (z.min()-5, z.max()+2),  # az - apex can be below deepest point
            (-1, 1),    # dx
            (-1, 1),    # dy
            (-1, 1),    # dz
        ]
        
        try:
            # Initial fit with all points
            result = least_squares(cone_residuals, initial_params, bounds=list(zip(*bounds)))
            
            if not result.success:
                print(f"Warning: Initial optimization did not converge: {result.message}")
            
            # Extract fitted parameters for outlier detection
            ax, ay, az, dx, dy, dz = result.x
            axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
            dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
            
            # Calculate residuals and remove outliers
            residuals = cone_residuals(result.x)
            residual_abs = np.abs(residuals)
            
            # Use robust statistics for outlier detection (STRICT THRESHOLDS)
            # Method 1: Modified Z-score using median absolute deviation (MAD)
            median_residual = np.median(residual_abs)
            mad = np.median(np.abs(residual_abs - median_residual))
            
            # More strict Modified Z-score threshold (reduced from 3.5 to 2.5)
            if mad > 0:
                modified_z_scores = 0.6745 * (residual_abs - median_residual) / mad
                outlier_threshold_mad = 2.5  # More strict threshold
                inlier_mask_mad = modified_z_scores < outlier_threshold_mad
            else:
                # Fallback: use IQR method if MAD is zero
                q1, q3 = np.percentile(residual_abs, [25, 75])
                iqr = q3 - q1
                if iqr > 0:
                    outlier_threshold_iqr = q3 + 1.0 * iqr  # More strict (reduced from 1.5)
                    inlier_mask_mad = residual_abs < outlier_threshold_iqr
                else:
                    # If both methods fail, keep all points
                    inlier_mask_mad = np.ones(len(residuals), dtype=bool)
            
            # Method 2: Absolute residual threshold (additional strict filter)
            # Remove points with residuals > 0.1mm (very strict for precision fitting)
            max_allowed_residual = 0.1  # mm
            inlier_mask_abs = residual_abs < max_allowed_residual
            
            # Combine both criteria (point must pass BOTH tests to be considered inlier)
            inlier_mask = inlier_mask_mad & inlier_mask_abs
            
            n_outliers = np.sum(~inlier_mask)
            n_inliers = np.sum(inlier_mask)
            n_outliers_mad = np.sum(~inlier_mask_mad)
            n_outliers_abs = np.sum(~inlier_mask_abs)
            
            print(f"  STRICT Outlier detection results:")
            print(f"    MAD/IQR method: {n_outliers_mad} outliers detected")
            print(f"    Absolute residual (>{max_allowed_residual}mm): {n_outliers_abs} outliers detected") 
            print(f"    Combined: {n_outliers} total outliers removed, {n_inliers} inliers retained")
            
            # Refit with inliers only if we have enough points and found outliers
            if n_inliers >= self.min_hole_points and n_outliers > 0:
                # Filter points to inliers only
                x_inliers = x[inlier_mask]
                y_inliers = y[inlier_mask]
                z_inliers = z[inlier_mask]
                
                # Redefine residual function for inliers
                def cone_residuals_inliers(params):
                    ax, ay, az, dx, dy, dz = params
                    
                    # Normalize axis direction
                    axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
                    if axis_norm < 1e-6:
                        return np.full(len(x_inliers), 1e6)
                    
                    dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
                    theta = fixed_theta
                    
                    # Vector from apex to each inlier point
                    px, py, pz = x_inliers - ax, y_inliers - ay, z_inliers - az
                    proj_length = px*dx + py*dy + pz*dz
                    valid_mask = proj_length > 0
                    
                    # Perpendicular distance from axis
                    perp_x = px - proj_length*dx
                    perp_y = py - proj_length*dy  
                    perp_z = pz - proj_length*dz
                    perp_dist = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
                    
                    # Expected radius at this height for cone
                    expected_radius = proj_length * np.tan(theta)
                    residuals = perp_dist - expected_radius
                    
                    # Penalize points on wrong side of apex
                    residuals[~valid_mask] = np.abs(residuals[~valid_mask]) * 10
                    
                    return residuals
                
                # Refit using inliers only
                result_refined = least_squares(cone_residuals_inliers, result.x, bounds=list(zip(*bounds)))
                
                if result_refined.success:
                    print(f"  Refined fit successful using {n_inliers} inlier points")
                    result = result_refined
                    # Calculate final residuals using all original points for RMSE
                    final_residuals = cone_residuals(result.x)
                else:
                    print(f"  Refined fit failed, using initial fit with all points")
                    final_residuals = residuals
            else:
                print(f"  No outlier removal: insufficient inliers ({n_inliers}) or no outliers found")
                final_residuals = residuals
            
            # Extract final fitted parameters
            ax, ay, az, dx, dy, dz = result.x
            
            # Normalize axis
            axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
            dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
            
            # Use fixed angle
            theta = fixed_theta
            
            # Store outlier information for visualization (ONLY from hole points used for fitting)
            outlier_coords = None
            inlier_coords = None
            if 'inlier_mask' in locals():
                outlier_mask = ~inlier_mask
                # These coordinates are ONLY from the hole points that were processed for cone fitting
                outlier_coords = {
                    'x': x[outlier_mask],
                    'y': y[outlier_mask], 
                    'z': z[outlier_mask]
                }
                inlier_coords = {
                    'x': x[inlier_mask],
                    'y': y[inlier_mask],
                    'z': z[inlier_mask]
                }
                
                # Verification: check that we're only using hole points
                total_outlier_inlier = len(outlier_coords['x']) + len(inlier_coords['x'])
                print(f"  🔍 Outlier/Inlier verification:")
                print(f"      Original hole points: {len(x)}")
                print(f"      Inliers: {len(inlier_coords['x'])}")
                print(f"      Outliers: {len(outlier_coords['x'])}")
                print(f"      Total (inliers + outliers): {total_outlier_inlier}")
                print(f"      ✅ Consistency check: {total_outlier_inlier == len(x)}")
                
            else:
                # If no outlier removal was applied, all hole points are inliers
                inlier_coords = {
                    'x': x.copy(),
                    'y': y.copy(),
                    'z': z.copy()
                }
                outlier_coords = {
                    'x': np.array([]),
                    'y': np.array([]),
                    'z': np.array([])
                }
                print(f"  🔍 No outlier removal applied - all {len(x)} hole points are inliers")
            
            fit_results = {
                'apex': np.array([ax, ay, az]),
                'axis': np.array([dx, dy, dz]),
                'half_angle_rad': theta,
                'half_angle_deg': np.degrees(theta),
                'included_angle_deg': np.degrees(2*theta),
                'rmse': np.sqrt(np.mean(final_residuals**2)),
                'success': result.success,
                'fixed_angle': True,  # Indicate angle was fixed, not fitted
                'n_outliers_removed': n_outliers,
                'n_inliers_used': n_inliers,
                'outlier_removal_applied': n_outliers > 0 and n_inliers >= self.min_hole_points,
                'outlier_coords': outlier_coords,
                'inlier_coords': inlier_coords,
                'original_coords': {'x': x, 'y': y, 'z': z}
            }
            
            print(f"Cone fitting results:")
            print(f"  Apex: ({ax:.3f}, {ay:.3f}, {az:.3f})")
            print(f"  Axis: ({dx:.3f}, {dy:.3f}, {dz:.3f})")
            print(f"  Half-angle: {np.degrees(theta):.1f}° (included: {np.degrees(2*theta):.1f}°) [FIXED]")
            print(f"  RMSE: {fit_results['rmse']:.4f} mm")
            if fit_results['outlier_removal_applied']:
                print(f"  Outlier removal: {fit_results['n_outliers_removed']} outliers removed from {len(x)} points")
            else:
                print(f"  Outlier removal: not applied ({fit_results['n_outliers_removed']} outliers detected)")
            
            return fit_results
            
        except Exception as e:
            print(f"Cone fitting (least squares) failed: {e}")
            raise
    
    def _fit_cone_slsqp(self, hole_points: Dict) -> Dict:
        """
        Alternative cone fitting using SLSQP optimizer with integrated outlier removal
        """
        
        x, y, z = hole_points['x'], hole_points['y'], hole_points['z']
        
        if len(x) < self.min_hole_points:
            raise ValueError(f"Insufficient points for cone fitting: {len(x)} < {self.min_hole_points}")
        
        # Initial guess (same as least squares method)
        hole_centroid = [np.mean(x), np.mean(y), np.mean(z)]
        hole_radius_estimate = np.sqrt(np.mean(x**2 + y**2))
        
        if hole_radius_estimate > 0:
            estimated_apex_offset = hole_radius_estimate / np.tan(self.expected_half_angle)
            initial_apex = [hole_centroid[0], hole_centroid[1], 
                           hole_centroid[2] - estimated_apex_offset]
        else:
            initial_apex = [hole_centroid[0], hole_centroid[1], z.min() - 0.5]
        
        initial_axis = [0, 0, 1]
        
        print(f"  Initial apex estimate (SLSQP): ({initial_apex[0]:.3f}, {initial_apex[1]:.3f}, {initial_apex[2]:.3f})")
        print(f"  Estimated hole radius: {hole_radius_estimate:.3f} mm")
        
        # Parameters: [ax, ay, az, dx, dy, dz]
        initial_params = np.array(initial_apex + initial_axis)
        fixed_theta = self.expected_half_angle
        
        # Keep track of outliers for integrated removal
        self.current_outlier_mask = np.ones(len(x), dtype=bool)  # Start with all points as inliers
        
        def objective_function_with_outlier_removal(params):
            """
            Objective function that includes outlier detection and removal
            Returns weighted sum of squared residuals
            """
            ax, ay, az, dx, dy, dz = params
            
            # Normalize axis direction
            axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
            if axis_norm < 1e-6:
                return 1e6  # Large penalty for degenerate case
            
            dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
            
            # Use only current inlier points
            x_current = x[self.current_outlier_mask]
            y_current = y[self.current_outlier_mask]
            z_current = z[self.current_outlier_mask]
            
            if len(x_current) < self.min_hole_points:
                return 1e6  # Penalty if too few inliers
            
            # Calculate residuals for current inliers
            px, py, pz = x_current - ax, y_current - ay, z_current - az
            proj_length = px*dx + py*dy + pz*dz
            
            # Only consider points projecting positively along axis
            valid_mask = proj_length > 0
            if np.sum(valid_mask) < self.min_hole_points // 2:
                return 1e6  # Penalty if too few valid projections
            
            # Perpendicular distance from axis
            perp_x = px - proj_length*dx
            perp_y = py - proj_length*dy
            perp_z = pz - proj_length*dz
            perp_dist = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
            
            # Expected radius at this height
            expected_radius = proj_length * np.tan(fixed_theta)
            residuals = perp_dist - expected_radius
            
            # Apply penalties for invalid projections
            residuals[~valid_mask] = np.abs(residuals[~valid_mask]) * 10
            
            # Robust cost function using Huber loss to naturally suppress outliers (STRICT)
            delta = 0.05  # Stricter Huber loss parameter (reduced from 0.1 to 0.05 mm)
            abs_residuals = np.abs(residuals)
            huber_loss = np.where(abs_residuals <= delta, 
                                  0.5 * residuals**2,
                                  delta * (abs_residuals - 0.5 * delta))
            
            return np.sum(huber_loss)
        
        def update_outlier_mask_callback(params):
            """
            Callback to periodically update the outlier mask during optimization
            """
            ax, ay, az, dx, dy, dz = params
            
            # Normalize axis
            axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
            if axis_norm < 1e-6:
                return
            dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
            
            # Calculate residuals for ALL points
            px, py, pz = x - ax, y - ay, z - az
            proj_length = px*dx + py*dy + pz*dz
            valid_proj_mask = proj_length > 0
            
            perp_x = px - proj_length*dx
            perp_y = py - proj_length*dy
            perp_z = pz - proj_length*dz
            perp_dist = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
            
            expected_radius = proj_length * np.tan(fixed_theta)
            residuals = perp_dist - expected_radius
            
            # Only consider points with valid projections for outlier detection
            valid_residuals = residuals[valid_proj_mask]
            
            if len(valid_residuals) > self.min_hole_points:
                # Use robust statistics for outlier detection (STRICT THRESHOLDS)
                median_res = np.median(np.abs(valid_residuals))
                mad = np.median(np.abs(np.abs(valid_residuals) - median_res))
                
                if mad > 0:
                    # More strict threshold (reduced from 2.5 to 2.0)
                    threshold_mad = median_res + 2.0 * mad
                    
                    # Additional strict absolute threshold
                    max_allowed_residual = 0.1  # mm - very strict
                    
                    # Update outlier mask - must pass both criteria
                    new_mask = np.ones(len(x), dtype=bool)
                    valid_indices = np.where(valid_proj_mask)[0]
                    
                    # Apply MAD-based threshold
                    mad_inliers = np.abs(valid_residuals) <= threshold_mad
                    # Apply absolute threshold  
                    abs_inliers = np.abs(valid_residuals) <= max_allowed_residual
                    # Combine both criteria
                    combined_inliers = mad_inliers & abs_inliers
                    
                    new_mask[valid_indices] = combined_inliers
                    new_mask[~valid_proj_mask] = False  # Invalid projections are outliers
                    
                    # Only update if we still have enough inliers
                    if np.sum(new_mask) >= self.min_hole_points:
                        self.current_outlier_mask = new_mask
        
        # Bounds for parameters [ax, ay, az, dx, dy, dz]
        bounds = [
            (x.min()-2, x.max()+2),  # ax
            (y.min()-2, y.max()+2),  # ay
            (z.min()-5, z.max()+2),  # az
            (-1, 1),    # dx
            (-1, 1),    # dy
            (-1, 1),    # dz
        ]
        
        # Constraint to keep axis normalized
        def axis_constraint(params):
            """Constraint to ensure axis is normalized"""
            dx, dy, dz = params[3], params[4], params[5]
            return dx**2 + dy**2 + dz**2 - 1.0
        
        constraints = {'type': 'eq', 'fun': axis_constraint}
        
        try:
            print(f"  Starting SLSQP optimization with {len(x)} points...")
            
            # Multiple optimization phases with outlier updates
            best_result = None
            best_cost = float('inf')
            
            for phase in range(3):  # Multiple phases to refine outlier detection
                
                # Run optimization
                result = minimize(objective_function_with_outlier_removal, 
                                initial_params,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints,
                                options={'maxiter': 800, 'ftol': 1e-12, 'disp': True})
                
                if result.success and result.fun < best_cost:
                    best_result = result
                    best_cost = result.fun
                
                # Update outlier mask for next phase
                if phase < 2:  # Don't update on last phase
                    update_outlier_mask_callback(result.x if result.success else initial_params)
                    n_inliers = np.sum(self.current_outlier_mask)
                    print(f"  Phase {phase+1}: {len(x) - n_inliers} outliers detected, {n_inliers} inliers")
                    
                    # Update initial guess for next phase
                    if result.success:
                        initial_params = result.x
            
            if best_result is None or not best_result.success:
                raise RuntimeError(f"SLSQP optimization failed: {best_result.message if best_result else 'No valid result'}")
            
            result = best_result
            
            # Final outlier statistics
            n_outliers_final = np.sum(~self.current_outlier_mask)
            n_inliers_final = np.sum(self.current_outlier_mask)
            
            # Extract final parameters
            ax, ay, az, dx, dy, dz = result.x
            
            # Ensure axis is normalized
            axis_norm = np.sqrt(dx**2 + dy**2 + dz**2)
            dx, dy, dz = dx/axis_norm, dy/axis_norm, dz/axis_norm
            
            # Calculate final RMSE using all original points
            px, py, pz = x - ax, y - ay, z - az
            proj_length = px*dx + py*dy + pz*dz
            valid_mask = proj_length > 0
            
            perp_x = px - proj_length*dx
            perp_y = py - proj_length*dy
            perp_z = pz - proj_length*dz
            perp_dist = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
            
            expected_radius = proj_length * np.tan(fixed_theta)
            final_residuals = perp_dist - expected_radius
            final_residuals[~valid_mask] = np.abs(final_residuals[~valid_mask]) * 10
            
            rmse = np.sqrt(np.mean(final_residuals**2))
            
            # Store outlier information for visualization (ONLY from hole points used for fitting)
            outlier_coords = None
            inlier_coords = None
            if hasattr(self, 'current_outlier_mask'):
                final_inlier_mask = self.current_outlier_mask
                outlier_mask = ~final_inlier_mask
                # These coordinates are ONLY from the hole points that were processed for cone fitting
                outlier_coords = {
                    'x': x[outlier_mask],
                    'y': y[outlier_mask], 
                    'z': z[outlier_mask]
                }
                inlier_coords = {
                    'x': x[final_inlier_mask],
                    'y': y[final_inlier_mask],
                    'z': z[final_inlier_mask]
                }
                
                # Verification: check that we're only using hole points
                total_outlier_inlier = len(outlier_coords['x']) + len(inlier_coords['x'])
                print(f"  🔍 SLSQP Outlier/Inlier verification:")
                print(f"      Original hole points: {len(x)}")
                print(f"      Inliers: {len(inlier_coords['x'])}")
                print(f"      Outliers: {len(outlier_coords['x'])}")
                print(f"      Total (inliers + outliers): {total_outlier_inlier}")
                print(f"      ✅ Consistency check: {total_outlier_inlier == len(x)}")
                
            else:
                # If no outlier removal was applied, all hole points are inliers
                inlier_coords = {
                    'x': x.copy(),
                    'y': y.copy(),
                    'z': z.copy()
                }
                outlier_coords = {
                    'x': np.array([]),
                    'y': np.array([]),
                    'z': np.array([])
                }
                print(f"  🔍 SLSQP: No outlier removal applied - all {len(x)} hole points are inliers")
            
            fit_results = {
                'apex': np.array([ax, ay, az]),
                'axis': np.array([dx, dy, dz]),
                'half_angle_rad': fixed_theta,
                'half_angle_deg': np.degrees(fixed_theta),
                'included_angle_deg': np.degrees(2*fixed_theta),
                'rmse': rmse,
                'success': result.success,
                'fixed_angle': True,
                'n_outliers_removed': n_outliers_final,
                'n_inliers_used': n_inliers_final,
                'outlier_removal_applied': n_outliers_final > 0,
                'outlier_coords': outlier_coords,
                'inlier_coords': inlier_coords,
                'original_coords': {'x': x, 'y': y, 'z': z},
                'optimizer': 'SLSQP',
                'optimization_cost': result.fun,
                'n_iterations': result.nit
            }
            
            print(f"Cone fitting results (SLSQP):")
            print(f"  Apex: ({ax:.3f}, {ay:.3f}, {az:.3f})")
            print(f"  Axis: ({dx:.3f}, {dy:.3f}, {dz:.3f})")
            print(f"  Half-angle: {np.degrees(fixed_theta):.1f}° (included: {np.degrees(2*fixed_theta):.1f}°) [FIXED]")
            print(f"  RMSE: {rmse:.4f} mm")
            print(f"  Optimization cost: {result.fun:.6f}")
            print(f"  Iterations: {result.nit}")
            if n_outliers_final > 0:
                print(f"  Outlier removal: {n_outliers_final} outliers removed from {len(x)} points")
            else:
                print(f"  Outlier removal: no outliers detected")
            
            # Clean up temporary attribute
            delattr(self, 'current_outlier_mask')
            
            return fit_results
            
        except Exception as e:
            # Clean up in case of error
            if hasattr(self, 'current_outlier_mask'):
                delattr(self, 'current_outlier_mask')
            print(f"Cone fitting (SLSQP) failed: {e}")
            raise
    
    def calculate_depth(self, surface_points: Dict, hole_points: Dict, cone_fit: Dict) -> Dict:
        """
        Calculate countersink depth based on fitted cone and estimated surface plane
        Improved precision by using consistent geometric calculations
        """
        # Use the estimated surface plane information
        surface_normal = self.results['surface_normal']
        surface_point = self.results['surface_point']
        
        # Get apex location and axis from cone fit
        apex = cone_fit['apex']
        axis = cone_fit['axis'].copy()
        half_angle = cone_fit['half_angle_rad']
        
        # Determine the correct axis direction more robustly
        # The axis should point from apex toward the surface/opening of the cone
        # Check which direction gives positive projection for the majority of hole points
        hole_points_array = np.column_stack([hole_points['x'], hole_points['y'], hole_points['z']])
        vectors_to_holes = hole_points_array - apex
        projections_positive = np.sum(np.dot(vectors_to_holes, axis) > 0)
        projections_negative = np.sum(np.dot(vectors_to_holes, axis) < 0)
        
        if projections_negative > projections_positive:
            axis = -axis
            print(f"  Flipped cone axis direction for consistency")
        
        # Calculate where cone intersects expected inner radius more precisely
        # For cone: r = h * tan(theta), where h is distance from apex along axis  
        # So: h = r / tan(theta)
        h_inner = self.expected_inner_radius / np.tan(half_angle)
        
        # Point on cone at inner radius (moving from apex along axis)
        # Use positive h_inner since we ensured axis points toward surface
        inner_point = apex + h_inner * axis
        
        # For more precise depth calculation, use the surface normal consistently
        # Calculate signed distances from surface plane (negative = below surface)
        apex_signed_distance = np.dot(apex - surface_point, surface_normal)
        inner_signed_distance = np.dot(inner_point - surface_point, surface_normal)

        # Apex should be the deepest point (most negative distance)
        # Inner radius intersection should be less deep (less negative)
        if apex_signed_distance > inner_signed_distance:
            # This suggests the axis direction or calculation is wrong
            print(f"  Warning: Apex appears above inner radius intersection")
            print(f"    Apex distance: {apex_signed_distance:.3f}, Inner distance: {inner_signed_distance:.3f}")
        
        # Convert to positive depth measurements (depth into material)
        apex_depth = -apex_signed_distance if apex_signed_distance < 0 else abs(apex_signed_distance)
        total_depth = -inner_signed_distance if inner_signed_distance < 0 else abs(inner_signed_distance)
        total_depth = apex_depth - h_inner  # More accurate total depth calculation

        # Geometric consistency check
        expected_depth_difference = h_inner * abs(np.dot(axis, surface_normal))
        actual_depth_difference = abs(apex_depth - total_depth)
        depth_consistency_ratio = actual_depth_difference / expected_depth_difference if expected_depth_difference > 0 else 0
        
        print(f"  Geometric consistency check:")
        print(f"    Expected depth difference: {expected_depth_difference:.3f} mm")
        print(f"    Actual depth difference: {actual_depth_difference:.3f} mm")
        print(f"    Consistency ratio: {depth_consistency_ratio:.3f} (should be ~1.0)")
        
        # Distance from inner radius point to apex along cone axis
        inner_to_apex_depth = h_inner
        
        depth_results = {
            'surface_normal': surface_normal.copy(),
            'surface_point': surface_point.copy(),
            'apex': apex.copy(),
            'inner_point': inner_point.copy(),
            'apex_signed_distance': apex_signed_distance,
            'inner_signed_distance': inner_signed_distance,
            'total_depth': total_depth,
            'apex_depth': apex_depth,
            'inner_to_apex_depth': inner_to_apex_depth,
            'expected_inner_radius': self.expected_inner_radius,
            'cone_axis_direction': axis.copy(),
            'depth_consistency_ratio': depth_consistency_ratio
        }
        
        print(f"\nDepth estimation results:")
        print(f"  Surface normal: ({surface_normal[0]:.3f}, {surface_normal[1]:.3f}, {surface_normal[2]:.3f})")
        print(f"  Surface point: ({surface_point[0]:.3f}, {surface_point[1]:.3f}, {surface_point[2]:.3f})")
        print(f"  Apex location: ({apex[0]:.3f}, {apex[1]:.3f}, {apex[2]:.3f}) mm")
        print(f"  Cone axis direction: ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
        print(f"  Apex depth from surface: {apex_depth:.3f} mm")
        print(f"  Inner radius intersection depth: {total_depth:.3f} mm")
        print(f"  Inner-to-apex distance: {inner_to_apex_depth:.3f} mm")
        print(f"  Expected inner radius: {depth_results['expected_inner_radius']:.3f} mm [FIXED]")
        
        return depth_results
    
    def estimate_depth_from_arrays(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
        """
        Estimate countersink depth directly from point cloud arrays
        This is the main method for direct array processing used by calibration scripts
        """
        print(f"Processing {len(x)} points for depth estimation")
        
        # Apply preprocessing transformations
        x, y, z = self.preprocess_point_cloud(x, y, z)
        
        # Apply noise filtering
        x, y, z, noise_mask = self.filter_noise(x, y, z)
        
        # Segment into surface and hole
        surface_points, hole_points = self.segment_surface_and_hole(x, y, z)
        
        if len(hole_points['x']) < self.min_hole_points:
            raise ValueError(f"Insufficient hole points for analysis: {len(hole_points['x'])}")
        
        # Fit cone to hole points
        cone_fit = self.fit_cone_to_points(hole_points)
        
        # Calculate depth
        depth_results = self.calculate_depth(surface_points, hole_points, cone_fit)
        
        # Create comprehensive results dictionary that matches the expected interface
        # results = {
        #     'total_points': len(x),
        #     'surface_points': len(surface_points['x']),
        #     'hole_points': len(hole_points['x']),
        #     'surface_normal': depth_results['surface_normal'],
        #     'surface_point': depth_results['surface_point'],
        #     'apex_location': depth_results['apex'],
        #     'cone_axis_direction': depth_results['cone_axis_direction'],
        #     'estimated_total_depth': depth_results['total_depth'],
        #     'estimated_apex_depth': depth_results['apex_depth'],
        #     'inner_to_apex_distance': depth_results['inner_to_apex_depth'],
        #     'expected_inner_radius': depth_results['expected_inner_radius'],
        #     'fit_rmse': cone_fit['rmse'],
        #     'consistency_ratio': depth_results['depth_consistency_ratio'],
        #     'cone_fit': cone_fit,
        #     'expected_params': {
        #         'csk_angle_deg': self.expected_csk_angle_deg,
        #         'inner_radius': self.expected_inner_radius
        #     }
        # }
        results = {
            'total_points': len(x),
            'segmentation': {
                'surface_points': len(surface_points['x']),
                'hole_points': len(hole_points['x'])
            },
            'surface_points': len(surface_points['x']),
            'hole_points': len(hole_points['x']),
            'surface_normal': depth_results['surface_normal'],
            'surface_point': depth_results['surface_point'],
            'apex_location': depth_results['apex'],
            'cone_axis_direction': depth_results['cone_axis_direction'],
            'estimated_total_depth': depth_results['total_depth'],
            'estimated_apex_depth': depth_results['apex_depth'],
            'inner_to_apex_distance': depth_results['inner_to_apex_depth'],
            'expected_inner_radius': depth_results['expected_inner_radius'],
            'fit_rmse': cone_fit['rmse'],
            'consistency_ratio': depth_results['depth_consistency_ratio'],
            'cone_fit': cone_fit,
            'depth': depth_results,
            'expected_params': {
                'csk_angle_deg': self.expected_csk_angle_deg,
                'inner_radius': self.expected_inner_radius
            }
        }
        
        self.results = results
        return results
    
    def estimate_depth(self, filename: str) -> Dict:
        """
        Main function to estimate countersink depth from point cloud file
        """
        print(f"Loading point cloud: {filename}")
        
        # Load point cloud
        x, y, z = self.load_point_cloud(filename)
        print(f"Loaded {len(x)} points")
        
        # Apply preprocessing transformations
        x, y, z = self.preprocess_point_cloud(x, y, z)
        
        # Apply noise filtering
        x, y, z, noise_mask = self.filter_noise(x, y, z)
        
        # Segment into surface and hole
        surface_points, hole_points = self.segment_surface_and_hole(x, y, z)
        
        if len(hole_points['x']) < self.min_hole_points:
            raise ValueError(f"Insufficient hole points for analysis: {len(hole_points['x'])}")
        
        # Fit cone to hole points
        cone_fit = self.fit_cone_to_points(hole_points)
        
        # Calculate depth
        depth_results = self.calculate_depth(surface_points, hole_points, cone_fit)
        
        # Combine all results
        # results = {
        #     'point_cloud_file': filename,
        #     'total_points': len(x),
        #     'segmentation': {
        #         'surface_points': len(surface_points['x']),
        #         'hole_points': len(hole_points['x'])
        #     },
        #     'cone_fit': cone_fit,
        #     'depth': depth_results,
        #     'expected_params': {
        #         'csk_angle_deg': self.expected_csk_angle_deg,
        #         'inner_radius': self.expected_inner_radius
        #     }
        # }
        results = {
            'total_points': len(x),
            'segmentation': {
                'surface_points': len(surface_points['x']),
                'hole_points': len(hole_points['x'])
            },
            'surface_points': len(surface_points['x']),
            'hole_points': len(hole_points['x']),
            'surface_normal': depth_results['surface_normal'],
            'surface_point': depth_results['surface_point'],
            'apex_location': depth_results['apex'],
            'cone_axis_direction': depth_results['cone_axis_direction'],
            'estimated_total_depth': depth_results['total_depth'],
            'estimated_apex_depth': depth_results['apex_depth'],
            'inner_to_apex_distance': depth_results['inner_to_apex_depth'],
            'expected_inner_radius': depth_results['expected_inner_radius'],
            'fit_rmse': cone_fit['rmse'],
            'consistency_ratio': depth_results['depth_consistency_ratio'],
            'cone_fit': cone_fit,
            'depth': depth_results,
            'expected_params': {
                'csk_angle_deg': self.expected_csk_angle_deg,
                'inner_radius': self.expected_inner_radius
            }
        }
        
        self.results = results
        return results
    
    def estimate_depth_with_global_plane(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                                       global_plane_params: Dict) -> Dict:
        """
        Estimate depth using global plane parameters instead of local plane fitting
        
        Args:
            x, y, z: Point cloud coordinates
            global_plane_params: Global plane parameters with keys 'normal', 'point', 'threshold'
            
        Returns:
            Dictionary with estimation results
        """
        print(f"Processing {len(x)} points for depth estimation with global plane")
        
        # Apply preprocessing (noise filtering)
        x_proc, y_proc, z_proc = self.preprocess_point_cloud(x, y, z)
        x_filtered, y_filtered, z_filtered, noise_mask = self.filter_noise(x_proc, y_proc, z_proc)
        
        # Use global plane parameters instead of estimating locally
        surface_normal = global_plane_params['normal']
        surface_point = global_plane_params['point']
        surface_threshold = global_plane_params['threshold']
        
        print(f"Using global surface normal: ({surface_normal[0]:.3f}, {surface_normal[1]:.3f}, {surface_normal[2]:.3f})")
        print(f"Using global surface point: ({surface_point[0]:.3f}, {surface_point[1]:.3f}, {surface_point[2]:.3f})")
        print(f"Using global surface thickness threshold: {surface_threshold:.3f} mm")
        
        # Segment surface and hole points using global plane
        points = np.column_stack([x_filtered, y_filtered, z_filtered])
        distances_to_plane = np.abs(np.dot(points - surface_point, surface_normal))
        
        # Points close to the plane are surface points
        surface_mask = distances_to_plane <= surface_threshold
        hole_mask = distances_to_plane > surface_threshold
        
        surface_points = {
            'x': x_filtered[surface_mask],
            'y': y_filtered[surface_mask], 
            'z': z_filtered[surface_mask]
        }
        
        hole_points = {
            'x': x_filtered[hole_mask],
            'y': y_filtered[hole_mask],
            'z': z_filtered[hole_mask]
        }
        
        print(f"Surface points: {len(surface_points['x'])}")
        print(f"Hole points: {len(hole_points['x'])}")
        
        if len(hole_points['x']) < self.min_hole_points:
            raise ValueError(f"Insufficient hole points: {len(hole_points['x'])} < {self.min_hole_points}")
        
        hole_distance_range = [np.min(distances_to_plane[hole_mask]), np.max(distances_to_plane[hole_mask])]
        print(f"Hole distance range: {hole_distance_range[0]:.3f} to {hole_distance_range[1]:.3f} mm from surface")
        
        hole_xy_range = [
            [np.min(hole_points['x']), np.max(hole_points['x'])],
            [np.min(hole_points['y']), np.max(hole_points['y'])]
        ]
        print(f"Hole xy range: x=[{hole_xy_range[0][0]:.3f}, {hole_xy_range[0][1]:.3f}], y=[{hole_xy_range[1][0]:.3f}, {hole_xy_range[1][1]:.3f}]")
        
        # Apply lateral filtering to hole points only (if enabled)
        if self.lateral_filter and len(hole_points['x']) > 0:
            hole_points_filtered, hole_lateral_mask = self._filter_hole_lateral_outliers(hole_points)
            hole_points = hole_points_filtered
            
            print(f"Hole points after lateral filtering: {len(hole_points['x'])}")
            
            if len(hole_points['x']) < self.min_hole_points:
                raise ValueError(f"Insufficient hole points after lateral filtering: {len(hole_points['x'])} < {self.min_hole_points}")
        
        # Apply z-direction filtering to hole points only (if enabled)
        if self.z_filter and len(hole_points['x']) > 0:
            hole_points_filtered, hole_z_mask = self._filter_hole_z_outliers(hole_points)
            hole_points = hole_points_filtered
            
            print(f"Hole points after z-direction filtering: {len(hole_points['x'])}")
            
            if len(hole_points['x']) < self.min_hole_points:
                raise ValueError(f"Insufficient hole points after z-direction filtering: {len(hole_points['x'])} < {self.min_hole_points}")
        
        # Store segmentation results for access by plotting
        self.surface_points = surface_points
        self.hole_points = hole_points
        
        # Fit cone to hole points
        cone_fit = self.fit_cone_to_points(hole_points)
        
        # Calculate depth using global plane parameters
        depth_results = self.calculate_depth_with_global_plane(
            surface_points, hole_points, cone_fit, global_plane_params
        )
        
        # Store results for plotting
        self.results = {
            'cone_fit': cone_fit,
            'surface_normal': surface_normal,
            'surface_point': surface_point,
            'surface_threshold': surface_threshold
        }
        
        return depth_results
    
    def calculate_depth_with_global_plane(self, surface_points: Dict, hole_points: Dict, 
                                        cone_fit: Dict, global_plane_params: Dict) -> Dict:
        """
        Calculate depth using global plane parameters
        
        Args:
            surface_points: Surface point dictionary with keys 'x', 'y', 'z'
            hole_points: Hole point dictionary with keys 'x', 'y', 'z'
            cone_fit: Cone fitting results with keys 'apex', 'axis', 'half_angle_rad', 'rmse'
            global_plane_params: Global plane parameters with keys 'normal', 'point', 'threshold'
            
        Returns:
            Dictionary with depth calculation results
        """
        surface_normal = global_plane_params['normal']
        surface_point = global_plane_params['point']
        
        # Get cone parameters
        apex = cone_fit['apex']
        axis = cone_fit['axis']
        half_angle = cone_fit['half_angle_rad']
        
        # Calculate apex depth from surface (distance along surface normal)
        apex_vector_from_surface = apex - surface_point
        apex_depth_from_surface = np.abs(np.dot(apex_vector_from_surface, surface_normal))
        
        # Calculate inner radius intersection point
        # Distance from apex to inner radius intersection along cone axis
        inner_radius_distance = self.expected_inner_radius / np.tan(half_angle)
        
        # Point at inner radius intersection
        inner_radius_point = apex + inner_radius_distance * (axis / np.linalg.norm(axis))
        
        # Calculate depth to inner radius intersection
        inner_vector_from_surface = inner_radius_point - surface_point
        inner_radius_depth = np.abs(np.dot(inner_vector_from_surface, surface_normal))

        # Calculate distance from inner radius intersection to apex
        inner_to_apex_distance = np.linalg.norm(apex - inner_radius_point)
        
        print(f"\nDepth estimation results:")
        print(f"  Surface normal: ({surface_normal[0]:.3f}, {surface_normal[1]:.3f}, {surface_normal[2]:.3f})")
        print(f"  Surface point: ({surface_point[0]:.3f}, {surface_point[1]:.3f}, {surface_point[2]:.3f})")
        print(f"  Apex location: ({apex[0]:.3f}, {apex[1]:.3f}, {apex[2]:.3f}) mm")
        print(f"  Cone axis direction: ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
        print(f"  Apex depth from surface: {apex_depth_from_surface:.3f} mm")
        print(f"  Inner radius intersection depth: {inner_radius_depth:.3f} mm")
        print(f"  Inner-to-apex distance: {inner_to_apex_distance:.3f} mm")
        print(f"  Expected inner radius: {self.expected_inner_radius:.3f} mm [FIXED]")
        
        return {
            'estimated_total_depth': inner_radius_depth,
            'estimated_apex_depth': apex_depth_from_surface,
            'fit_rmse': cone_fit['rmse'],
            'total_points': len(surface_points['x']) + len(hole_points['x']),
            'surface_points': len(surface_points['x']),
            'hole_points': len(hole_points['x']),
            'cone_apex': apex,
            'cone_axis': axis,
            'cone_half_angle': half_angle,
            'surface_normal': surface_normal,
            'surface_point': surface_point,
            'inner_radius_intersection_point': inner_radius_point,
            'inner_to_apex_distance': inner_to_apex_distance,
            'optimization_cost': cone_fit.get('cost', np.nan),
            'optimization_iterations': cone_fit.get('iterations', 0)
        }
    
    def plot_fitted_cone(self, ax, apex, axis, half_angle, cone_height=0.965, n_circles=15, n_points_per_circle=30):
        """
        Plot the fitted cone surface with proper geometry
        
        Parameters:
        - ax: 3D matplotlib axis
        - apex: cone apex position [x, y, z]
        - axis: cone axis direction [dx, dy, dz] (normalized)
        - half_angle: cone half angle in radians
        - cone_height: height of cone to plot from apex
        - n_circles: number of circular cross-sections to draw
        - n_points_per_circle: number of points per circle
        """
        
        # Generate cone surface points
        heights = np.linspace(0.01, cone_height, n_circles)  # Start slightly above apex
        theta_circle = np.linspace(0, 2*np.pi, n_points_per_circle)
        
        cone_x, cone_y, cone_z = [], [], []
        
        for h in heights:
            radius = h * np.tan(half_angle)
            
            # Create a circle at this height
            circle_points = np.zeros((n_points_per_circle, 3))
            
            # Generate circle in local coordinate system
            for i, theta in enumerate(theta_circle):
                # Circle in xy-plane of local coordinate system
                local_x = radius * np.cos(theta)
                local_y = radius * np.sin(theta)
                local_z = 0
                
                # Transform to world coordinates
                # We need to create a coordinate system where axis is the z-direction
                # Use Gram-Schmidt to create orthonormal basis
                z_axis = axis / np.linalg.norm(axis)
                
                # Find a vector not parallel to z_axis
                if abs(z_axis[2]) < 0.9:
                    temp_vec = np.array([0, 0, 1])
                else:
                    temp_vec = np.array([1, 0, 0])
                
                # Create orthonormal basis
                x_axis = np.cross(temp_vec, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                
                # Transform local coordinates to world coordinates
                world_point = (apex + h * z_axis + 
                             local_x * x_axis + 
                             local_y * y_axis)
                
                circle_points[i] = world_point
            
            cone_x.extend(circle_points[:, 0])
            cone_y.extend(circle_points[:, 1])
            cone_z.extend(circle_points[:, 2])
        
        # Plot cone surface as wireframe
        cone_x = np.array(cone_x).reshape(n_circles, n_points_per_circle)
        cone_y = np.array(cone_y).reshape(n_circles, n_points_per_circle)
        cone_z = np.array(cone_z).reshape(n_circles, n_points_per_circle)
        
        # Plot circular cross-sections
        for i in range(n_circles):
            ax.plot(cone_x[i, :], cone_y[i, :], cone_z[i, :], 'g-', alpha=0.4, linewidth=1)
        
        # Plot meridional lines (lines from apex to edge)
        n_meridians = 8
        meridian_indices = np.linspace(0, n_points_per_circle-1, n_meridians, dtype=int)
        
        for idx in meridian_indices:
            # Line from apex to the edge
            line_x = np.concatenate([[apex[0]], cone_x[:, idx]])
            line_y = np.concatenate([[apex[1]], cone_y[:, idx]])
            line_z = np.concatenate([[apex[2]], cone_z[:, idx]])
            ax.plot(line_x, line_y, line_z, 'g-', alpha=0.5, linewidth=1)
        
        # Mark the apex prominently
        ax.scatter([apex[0]], [apex[1]], [apex[2]], c='red', s=100, marker='*', 
                  edgecolors='black', linewidth=2, label='Cone Apex', zorder=10)
        
        # Draw cone axis line
        axis_end = apex + cone_height * axis
        ax.plot([apex[0], axis_end[0]], [apex[1], axis_end[1]], [apex[2], axis_end[2]], 
                'r--', linewidth=2, alpha=0.8, label='Cone Axis')
        
        return ax
    
    def plot_results(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, cone_height: float = 0.965):
        """Plot segmentation and fitting results with outlier visualization"""
        
        # Apply same preprocessing as in estimation
        x_proc, y_proc, z_proc = self.preprocess_point_cloud(x, y, z)
        x_filtered, y_filtered, z_filtered, noise_mask = self.filter_noise(x_proc, y_proc, z_proc)
        
        surface_points, hole_points = self.segment_surface_and_hole(x_filtered, y_filtered, z_filtered)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Original point cloud with noise filtering visualization
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        if self.noise_filter:
            # Show noise points and filtered points
            noise_points_mask = ~noise_mask
            if np.any(noise_points_mask):
                ax1.scatter(x_proc[noise_points_mask], y_proc[noise_points_mask], z_proc[noise_points_mask], 
                           c='red', s=3, alpha=0.8, label=f'Noise ({np.sum(noise_points_mask)})')
            ax1.scatter(x_filtered, y_filtered, z_filtered, c=z_filtered, cmap='viridis', s=1, alpha=0.6, label='Filtered')
            ax1.set_title('Noise Filtering Results')
        else:
            ax1.scatter(x_filtered, y_filtered, z_filtered, c=z_filtered, cmap='viridis', s=1, alpha=0.6)
            ax1.set_title('Original Point Cloud')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.legend()
        
        # 2. Segmented point cloud
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(surface_points['x'], surface_points['y'], surface_points['z'], 
                   c='blue', s=1, alpha=0.6, label='Surface')
        ax2.scatter(hole_points['x'], hole_points['y'], hole_points['z'], 
                   c='red', s=1, alpha=0.8, label='Hole')
        ax2.set_title('Segmented Points')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        ax2.legend()
        
        # 3. Fitted cone with outlier visualization (HOLE POINTS ONLY)
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        
        if 'cone_fit' in self.results:
            cone_fit = self.results['cone_fit']
            
            # Plot inliers and outliers if available (these should ONLY be from hole points)
            if cone_fit.get('outlier_coords') is not None and cone_fit.get('inlier_coords') is not None:
                outlier_coords = cone_fit['outlier_coords']
                inlier_coords = cone_fit['inlier_coords']
                
                # Verify that these are actually hole points by checking they're all from the hole region
                print(f"Debug: Inlier coords count: {len(inlier_coords['x'])}")
                print(f"Debug: Outlier coords count: {len(outlier_coords['x'])}")
                print(f"Debug: Total hole points: {len(hole_points['x'])}")
                
                # Plot inliers in green (these are hole points that fit the cone well)
                if len(inlier_coords['x']) > 0:
                    ax3.scatter(inlier_coords['x'], inlier_coords['y'], inlier_coords['z'], 
                               c='green', s=2, alpha=0.8, label=f'Hole Inliers ({len(inlier_coords["x"])})')
                
                # Plot outliers in red (these are hole points that don't fit the cone well)
                if len(outlier_coords['x']) > 0:
                    ax3.scatter(outlier_coords['x'], outlier_coords['y'], outlier_coords['z'], 
                               c='red', s=4, alpha=0.9, label=f'Hole Outliers ({len(outlier_coords["x"])})')
            else:
                # Fallback to original hole points if outlier info not available
                ax3.scatter(hole_points['x'], hole_points['y'], hole_points['z'], 
                           c='red', s=1, alpha=0.6, label=f'Hole points ({len(hole_points["x"])})')
            
            apex = cone_fit['apex']
            axis = cone_fit['axis'] 
            half_angle = cone_fit['half_angle_rad']
            
            # Use the detailed cone visualization
            self.plot_fitted_cone(ax3, apex, axis, half_angle, cone_height)
            
            ax3.set_title('Cone Fit: Hole Points Only (Inliers vs Outliers)')
            ax3.legend()
        else:
            ax3.scatter(hole_points['x'], hole_points['y'], hole_points['z'], 
                       c='red', s=1, alpha=0.6, label='Hole points')
            ax3.set_title('Hole Points (No Cone Fit Available)')
        
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        
        # 4. Outlier analysis histogram
        ax4 = fig.add_subplot(2, 3, 4)
        if 'cone_fit' in self.results and self.results['cone_fit'].get('outlier_coords') is not None:
            cone_fit = self.results['cone_fit']
            outlier_coords = cone_fit['outlier_coords']
            inlier_coords = cone_fit['inlier_coords']
            
            if len(outlier_coords['x']) > 0 and len(inlier_coords['x']) > 0:
                # Plot Z-distribution comparison
                ax4.hist(inlier_coords['z'], bins=30, alpha=0.6, label='Inliers', color='green', density=True)
                ax4.hist(outlier_coords['z'], bins=30, alpha=0.6, label='Outliers', color='red', density=True)
                ax4.set_xlabel('Z coordinate (mm)')
                ax4.set_ylabel('Density')
                ax4.set_title('Z-Distribution: Inliers vs Outliers')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # 5. Residual analysis
        ax5 = fig.add_subplot(2, 3, 5)
        if 'cone_fit' in self.results:
            cone_fit = self.results['cone_fit']
            # Add residual analysis here if residuals are stored
            ax5.text(0.5, 0.5, 'Residual Analysis\n(To be implemented)', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Residual Analysis')
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = "ANALYSIS SUMMARY\n"
        summary_text += "="*40 + "\n"
        summary_text += f"Original points: {len(x):,}\n"
        
        if self.noise_filter:
            noise_removed = len(x_proc) - len(x_filtered)
            summary_text += f"Noise points removed: {noise_removed:,}\n"
        
        summary_text += f"Surface points: {len(surface_points['x']):,}\n"
        summary_text += f"Hole points: {len(hole_points['x']):,}\n\n"
        
        if 'cone_fit' in self.results:
            cone_fit = self.results['cone_fit']
            summary_text += "CONE FITTING:\n"
            summary_text += f"Half-angle: {cone_fit['half_angle_deg']:.1f}°\n"
            summary_text += f"RMSE: {cone_fit['rmse']:.4f} mm\n"
            summary_text += f"Outliers removed: {cone_fit.get('n_outliers_removed', 0)}\n"
            summary_text += f"Inliers used: {cone_fit.get('n_inliers_used', 0)}\n\n"
        
        if 'depth' in self.results:
            depth = self.results['depth']
            summary_text += "DEPTH ESTIMATION:\n"
            summary_text += f"Total depth: {depth['total_depth']:.3f} mm\n"
            summary_text += f"Apex depth: {depth['apex_depth']:.3f} mm\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax3.set_title(f'Fitted Cone (height={cone_height:.3f}mm)')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Estimate countersink depth from 3D point cloud")
    parser.add_argument('filename', nargs='?', default='countersink_3d_scan.pcd', 
                       help='Point cloud file (.pcd or .csv)')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    parser.add_argument('--csk-angle', type=float, default=100.0, 
                       help='Expected countersink angle (degrees)')
    parser.add_argument('--inner-radius', type=float, default=1.2446,
                       help='Expected inner radius (mm)')
    parser.add_argument('--optimizer', type=str, default='slsqp', 
                       choices=['least_squares', 'slsqp'],
                       help='Optimization method: least_squares or slsqp')
    parser.add_argument('--cone-height', type=float, default=0.965,
                       help='Height of cone to visualize (mm)')
    parser.add_argument('--flip-z', action='store_true', 
                       help='Flip z-coordinates (multiply by -1)')
    parser.add_argument('--meters-to-mm', action='store_true',
                       help='Convert coordinates from meters to millimeters')
    parser.add_argument('--no-noise-filter', action='store_true',
                       help='Disable noise filtering')
    parser.add_argument('--noise-neighbors', type=int, default=100,
                       help='Minimum neighbors for noise filter (default: 100)')
    parser.add_argument('--noise-radius', type=float, default=0.5,
                       help='Search radius for noise filter in mm (default: 0.5)')
    parser.add_argument('--no-lateral-filter', action='store_true',
                       help='Disable lateral distance filtering from hole center')
    parser.add_argument('--lateral-filter-threshold', type=float, default=3.0,
                       help='Maximum lateral distance from hole center in mm (default: 3.0)')
    parser.add_argument('--visualize-lateral-filter', action='store_true',
                       help='Show visualization of lateral filtering process')
    parser.add_argument('--no-z-filter', action='store_true',
                       help='Disable z-direction filtering for hole points')
    parser.add_argument('--z-filter-threshold', type=float, default=0.6,
                       help='Maximum z distance from median z in mm (default: 0.6)')
    parser.add_argument('--ransac-sample-points', type=int, default=10,
                       help='Number of points to sample for RANSAC plane fitting (default: 10)')
    parser.add_argument('--outlier-method', type=str, default='percentile', 
                       choices=['percentile', 'median_filter'],
                       help='Outlier removal method: percentile (remove top X%) or median_filter (remove costs > X*median)')
    parser.add_argument('--outlier-threshold', type=float, default=0.02,
                       help='Outlier threshold: for percentile=fraction to remove (0.02=2%), for median_filter=multiplier (3.0=3x median)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible RANSAC results (default: 42)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found")
        return
    
    # Create estimator
    estimator = CountersinkDepthEstimator(
        expected_csk_angle_deg=args.csk_angle,
        expected_inner_radius=args.inner_radius,
        optimizer=args.optimizer,
        flip_z=args.flip_z,
        meters_to_mm=args.meters_to_mm,
        noise_filter=not args.no_noise_filter,
        noise_neighbors=args.noise_neighbors,
        noise_radius=args.noise_radius,
        lateral_filter=not args.no_lateral_filter,
        lateral_filter_threshold=args.lateral_filter_threshold,
        visualize_lateral_filter=args.visualize_lateral_filter,
        z_filter=not args.no_z_filter,
        z_filter_threshold=args.z_filter_threshold,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold,
        ransac_sample_points=args.ransac_sample_points,
        random_seed=args.random_seed
    )
    
    try:
        # Estimate depth
        results = estimator.estimate_depth(args.filename)
        
        # Print summary
        print("\n" + "="*50)
        print("COUNTERSINK DEPTH ESTIMATION SUMMARY")
        print("="*50)
        print(f"File: {args.filename}")
        print(f"Total depth: {results['depth']['total_depth']:.3f} mm")
        print(f"Fitted cone angle: {results['cone_fit']['included_angle_deg']:.1f}°")
        print(f"Fit RMSE: {results['cone_fit']['rmse']:.4f} mm")
        
        # Show plots if requested
        if args.plot:
            x, y, z = estimator.load_point_cloud(args.filename)
            estimator.plot_results(x, y, z, cone_height=args.cone_height)
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        return

if __name__ == "__main__":
    main()