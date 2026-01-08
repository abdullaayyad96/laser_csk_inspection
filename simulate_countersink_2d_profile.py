import numpy as np
from typing import Tuple, Optional

import matplotlib.pyplot as plt

class CountersinkLaserScanner:
    def __init__(self, 
                 csk_angle: float = 100.0,  # degrees
                 scanner_range: float = 10.0,  # mm
                 x_offset: float = 0.0,  # mm
                 y_offset: float = 0.0,  # mm
                 standoff: float = 5.0,  # mm
                 inner_radius: float = 2.0,  # mm
                 outer_rim_depth_variation: float = 0.0,  # mm
                 hole_depth: float = 3.0,  # mm
                 inclination_angle: float = 0.0,  # degrees
                 noise_power: float = 0.01,  # mm
                 dropout_factor: float = 0.0,  # 0-1 probability
                 min_angle_deg: float = -30.0,  # degrees - minimum scanning angle
                 max_angle_deg: float = 30.0):  # degrees - maximum scanning angle  # 0-1 probability
        
        self.csk_angle = np.radians(csk_angle)
        self.scanner_range = scanner_range
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.standoff = standoff
        self.inner_radius = inner_radius
        self.outer_rim_depth_variation = outer_rim_depth_variation
        self.hole_depth = hole_depth
        self.inclination_angle = np.radians(inclination_angle)
        self.noise_power = noise_power
        self.dropout_factor = dropout_factor
        self.min_angle_deg = min_angle_deg
        self.max_angle_deg = max_angle_deg
        
        # Calculate outer radius based on countersink geometry
        self.outer_radius = inner_radius + hole_depth * np.tan(self.csk_angle / 2)
    
    def generate_scan_points(self, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate laser scan points across the lateral range"""
        # Generate scan points along x-axis (lateral scan)
        x_scan = np.linspace(-self.scanner_range/2, self.scanner_range/2, num_points)
        y_scan = np.zeros_like(x_scan)  # 2D scanner along x-axis
        
        # Apply scanner offset
        x_scan += self.x_offset
        y_scan += self.y_offset
        
        return x_scan, y_scan, np.zeros_like(x_scan)
    
    def calculate_surface_height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the height of the countersink surface at given x,y coordinates"""
        # Distance from hole center
        r = np.sqrt(x**2 + y**2)
        
        # Initialize height array (flat surface level = 0)
        height = np.zeros_like(r)
        
        # Points inside inner radius - flat bottom
        inner_mask = r <= self.inner_radius
        height[inner_mask] = -self.hole_depth
        
        # Points in countersink region
        csk_mask = (r > self.inner_radius) & (r <= self.outer_radius)
        if np.any(csk_mask):
            # Linear interpolation for countersink slope
            slope_distance = r[csk_mask] - self.inner_radius
            max_slope_distance = self.outer_radius - self.inner_radius
            height[csk_mask] = -self.hole_depth + (slope_distance / max_slope_distance) * self.hole_depth
            
            # Add outer rim depth variation (non-linear)
            if self.outer_rim_depth_variation > 0:
                height[csk_mask] -= self.outer_rim_depth_variation
        
        # Points outside countersink remain at surface level (height = 0)
        
        return height
    
    def apply_inclination(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply inclination angle to the surface - rotates both x and z coordinates"""
        if self.inclination_angle != 0:
            # Rotation matrix for rotation about y-axis
            cos_theta = np.cos(self.inclination_angle)
            sin_theta = np.sin(self.inclination_angle)
            
            # Apply rotation: [x', z'] = [cos(θ) sin(θ); -sin(θ) cos(θ)] * [x; z]
            x_rotated = cos_theta * x + sin_theta * z
            z_rotated = -sin_theta * x + cos_theta * z
            
            return x_rotated, z_rotated
        return x, z
    
    def simulate_laser_measurement(self, x_scan: np.ndarray, y_scan: np.ndarray) -> np.ndarray:
        """Simulate laser distance measurements"""
        # Distance from hole center (using original coordinates for hole geometry)
        r = np.sqrt(x_scan**2 + y_scan**2)
        
        # Calculate surface height
        surface_height = self.calculate_surface_height(x_scan, y_scan)
        
        # Apply inclination (this rotates the coordinate system)
        x_rotated, z_rotated = self.apply_inclination(x_scan, y_scan, surface_height)
        
        # Calculate laser measurement (standoff - rotated surface height)
        laser_distance = self.standoff - z_rotated
        
        # Set inner hole region to NaN (no laser return - see through)
        # Use original coordinates for hole mask since geometry doesn't change
        inner_hole_mask = r <= self.inner_radius
        laser_distance[inner_hole_mask] = np.nan
        
        # Add noise to valid measurements only
        if self.noise_power > 0:
            valid_mask = ~np.isnan(laser_distance)
            noise = np.random.normal(0, self.noise_power, laser_distance.shape)
            laser_distance[valid_mask] += noise[valid_mask]
        
        return x_rotated, y_scan, laser_distance

    def uniform_angular_sampling(self, x_rotated: np.ndarray, y_scan: np.ndarray, 
                                laser_distance: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample the data to uniform angular spacing based on theta = arctan(x/z)
        where x is lateral position and z is laser distance (both in laser frame)
        Preserves NaN values when no valid points are available for interpolation
        """
        # Calculate angles for ALL points (including NaN distances)
        # We'll handle NaN separately
        all_angles = np.full_like(laser_distance, np.nan)
        valid_mask = ~np.isnan(laser_distance) & (laser_distance > 0)
        
        if not np.any(valid_mask):
            # No valid points, return arrays filled with NaN
            return np.full(num_points, np.nan), np.full(num_points, np.nan), np.full(num_points, np.nan)
        
        # Calculate angles only for valid points
        x_valid = x_rotated[valid_mask]
        y_valid = y_scan[valid_mask]  
        dist_valid = laser_distance[valid_mask]
        angles_valid = np.degrees(np.arctan(x_valid / dist_valid))
        
        # Create uniform angular grid
        angle_min = max(self.min_angle_deg, angles_valid.min())
        angle_max = min(self.max_angle_deg, angles_valid.max())
        
        if angle_min >= angle_max:
            # Invalid range, return arrays filled with NaN
            return np.full(num_points, np.nan), np.full(num_points, np.nan), np.full(num_points, np.nan)
        
        uniform_angles = np.linspace(angle_min, angle_max, num_points)
        
        # Initialize output arrays with NaN
        x_uniform = np.full(num_points, np.nan)
        y_uniform = np.full(num_points, np.nan)
        dist_uniform = np.full(num_points, np.nan)
        
        # Sort the valid data by angle for proper interpolation
        sort_idx = np.argsort(angles_valid)
        angles_sorted = angles_valid[sort_idx]
        x_sorted = x_valid[sort_idx]
        y_sorted = y_valid[sort_idx] 
        dist_sorted = dist_valid[sort_idx]
        
        # For each uniform angle, check if there are valid points nearby for interpolation
        # Define a tolerance for "nearby" - we won't interpolate across large gaps
        angle_tolerance = (angle_max - angle_min) / (len(angles_sorted) - 1) * 2  # 2x average spacing
        
        for i, target_angle in enumerate(uniform_angles):
            # Find the closest valid angles on either side
            below_mask = angles_sorted <= target_angle
            above_mask = angles_sorted >= target_angle
            
            if np.any(below_mask) and np.any(above_mask):
                # Get closest valid points on both sides
                below_idx = np.where(below_mask)[0][-1] if np.any(below_mask) else None
                above_idx = np.where(above_mask)[0][0] if np.any(above_mask) else None
                
                # Check if the gap is small enough to interpolate
                valid_to_interpolate = False
                
                if below_idx is not None and above_idx is not None:
                    # Both sides available
                    gap_size = angles_sorted[above_idx] - angles_sorted[below_idx]
                    if gap_size <= angle_tolerance:
                        valid_to_interpolate = True
                elif below_idx is not None and above_idx is not None and below_idx == above_idx:
                    # Exact match
                    valid_to_interpolate = True
                
                if valid_to_interpolate:
                    # Safe to interpolate
                    x_uniform[i] = np.interp(target_angle, angles_sorted, x_sorted)
                    y_uniform[i] = np.interp(target_angle, angles_sorted, y_sorted)
                    dist_uniform[i] = np.interp(target_angle, angles_sorted, dist_sorted)
        
        # Ensure consistency: if any coordinate is NaN, make all NaN
        nan_mask = np.isnan(x_uniform) | np.isnan(y_uniform) | np.isnan(dist_uniform)
        x_uniform[nan_mask] = np.nan
        y_uniform[nan_mask] = np.nan
        dist_uniform[nan_mask] = np.nan
        
        return x_uniform, y_uniform, dist_uniform
    
    def scan(self, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform complete scan simulation with uniform angular sampling"""
        # Generate initial dense sampling
        initial_points = max(num_points * 4, 1000)  # Oversample initially for better interpolation
        x_scan, y_scan, _ = self.generate_scan_points(initial_points)
        x_rotated, y_scan, distances = self.simulate_laser_measurement(x_scan, y_scan)

        # Resample to uniform angular spacing
        x_uniform, y_uniform, dist_uniform = self.uniform_angular_sampling(
            x_rotated, y_scan, distances, num_points)
        
        # Apply dropout to final uniform samples
        if self.dropout_factor > 0:
            valid_mask = ~np.isnan(dist_uniform)
            dropout_mask = np.random.random(dist_uniform.shape) < self.dropout_factor
            dropout_points = valid_mask & dropout_mask
            
            # Set dropped points to NaN for all coordinates
            x_uniform[dropout_points] = np.nan
            y_uniform[dropout_points] = np.nan
            dist_uniform[dropout_points] = np.nan
        
        return x_uniform, y_uniform, dist_uniform
    
    def plot_results(self, x_scan: np.ndarray, y_scan: np.ndarray, distances: np.ndarray):
        """Plot scan results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot distance measurements
        valid_mask = ~np.isnan(distances)
        ax1.scatter(x_scan[valid_mask], distances[valid_mask], c='b', marker='.', s=2)
        ax1.set_xlabel('X Position (mm)')
        ax1.set_ylabel('Laser Distance (mm)')
        ax1.set_title('Laser Scanner Distance Measurements')
        ax1.grid(True)
        
        # Plot reconstructed surface profile
        surface_height = self.standoff - distances
        ax2.plot(x_scan[valid_mask], surface_height[valid_mask], 'r.-', markersize=2)
        ax2.set_xlabel('X Position (mm)')
        ax2.set_ylabel('Surface Height (mm)')
        ax2.set_title('Reconstructed Surface Profile')
        ax2.grid(True)
        ax2.invert_yaxis()  # Invert to show hole going down
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create scanner with example parameters
    scanner = CountersinkLaserScanner(
        csk_angle=100.0,           # 100 degree countersink
        scanner_range=15.0,        # 15mm scan range
        x_offset=0.0,              # 0.0mm offset in x
        y_offset=0.0,              # 0.0mm offset in y
        standoff=85.0,             # 85mm standoff distance
        inner_radius=1.25,          # 1.25mm inner radius (larger to create void)
        outer_rim_depth_variation=0.1,  # 0.1mm rim variation
        hole_depth=0.965,          # 0.965mm hole depth
        inclination_angle=0.0,    # 15 degree inclination
        noise_power=0.004,         # 0.004mm noise
        dropout_factor=0.15,       # 15% dropout rate (higher to test NaN handling)
        min_angle_deg=-30.0,       # -30 degree minimum angle
        max_angle_deg=30.0         # 30 degree maximum angle
    )
    
    # Perform scan
    x, y, distances = scanner.scan(num_points=2056)

    # Plot results
    scanner.plot_results(x, y, distances)
    
    # Print some statistics
    valid_distances = distances[~np.isnan(distances)]
    print(f"Scan Statistics:")
    print(f"Total points: {len(distances)}")
    print(f"Valid points: {len(valid_distances)}")
    print(f"Dropout rate: {(len(distances) - len(valid_distances)) / len(distances) * 100:.1f}%")
    print(f"Distance range: {valid_distances.min():.2f} - {valid_distances.max():.2f} mm")