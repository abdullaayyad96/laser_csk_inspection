import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simulate_countersink_2d_profile import CountersinkLaserScanner

class Countersink3DScanner:
    def __init__(self,
                 # 2D scanner parameters
                 csk_angle: float = 100.0,
                 scanner_range: float = 10.0,
                 x_offset: float = 0.0,
                 standoff: float = 85.0,
                 inner_radius: float = 1.25,
                 outer_rim_depth_variation: float = 0.2,
                 hole_depth: float = 0.965,
                 inclination_angle: float = 0.0,
                 noise_power: float = 0.005,
                 dropout_factor: float = 0.05,
                 min_angle_deg: float = -30.0,
                 max_angle_deg: float = 30.0,
                 # 3D scanning parameters
                 y_min: float = -7.5,  # mm - minimum y offset
                 y_max: float = 7.5,   # mm - maximum y offset
                 y_steps: int = 30,    # number of y positions
                 points_per_scan: int = 500,  # points per 2D scan
                 x_inclination_angle_deg: float = 0.0,  # rotation around x-axis for full 3D scan
                 y_offset_noise: float = 0.05):  # mm - noise in y positioning
        
        # Store 3D scanning parameters
        self.y_min = y_min
        self.y_max = y_max
        self.y_steps = y_steps
        self.points_per_scan = points_per_scan
        self.x_inclination_angle_deg = x_inclination_angle_deg
        self.x_inclination_angle_rad = np.radians(x_inclination_angle_deg)
        self.y_offset_noise = y_offset_noise
        
        # Store 2D scanner parameters (we'll create individual scanners for each y position)
        self.scanner_params = {
            'csk_angle': csk_angle,
            'scanner_range': scanner_range,
            'x_offset': x_offset,
            'standoff': standoff,
            'inner_radius': inner_radius,
            'outer_rim_depth_variation': outer_rim_depth_variation,
            'hole_depth': hole_depth,
            'inclination_angle': inclination_angle,
            'noise_power': noise_power,
            'dropout_factor': dropout_factor,
            'min_angle_deg': min_angle_deg,
            'max_angle_deg': max_angle_deg
        }
    
    def generate_y_positions(self) -> np.ndarray:
        """Generate y positions for scanning with optional noise"""
        # Generate uniform y positions
        y_positions = np.linspace(self.y_min, self.y_max, self.y_steps)
        
        # Add positioning noise if specified
        if self.y_offset_noise > 0:
            noise = np.random.normal(0, self.y_offset_noise, size=y_positions.shape)
            y_positions += noise
        
        return y_positions
    
    def perform_2d_scan(self, y_offset: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform a single 2D scan at the specified y offset"""
        # Create scanner with the specified y offset
        scanner = CountersinkLaserScanner(
            y_offset=y_offset,
            **self.scanner_params
        )
        
        # Perform the scan
        x, y, distances = scanner.scan(num_points=self.points_per_scan)
        
        return x, y, distances
    
    def apply_x_rotation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply rotation around x-axis to the entire 3D point cloud"""
        if self.x_inclination_angle_rad != 0:
            # Rotation matrix for rotation about x-axis
            cos_theta = np.cos(self.x_inclination_angle_rad)
            sin_theta = np.sin(self.x_inclination_angle_rad)
            
            # Apply rotation: y' = cos(θ)*y - sin(θ)*z, z' = sin(θ)*y + cos(θ)*z
            y_rotated = cos_theta * y - sin_theta * z
            z_rotated = sin_theta * y + cos_theta * z
            
            return x, y_rotated, z_rotated
        
        return x, y, z
    
    def scan_3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform full 3D scan by combining multiple 2D scans"""
        print(f"Starting 3D scan with {self.y_steps} positions...")
        
        # Generate y positions
        y_positions = self.generate_y_positions()
        
        # Lists to collect all points
        all_x = []
        all_y = []
        all_z = []
        
        # Perform 2D scan at each y position
        for i, y_pos in enumerate(y_positions):
            print(f"Scanning position {i+1}/{self.y_steps} at y={y_pos:.3f} mm")
            
            # Perform 2D scan
            x_2d, y_2d, distances = self.perform_2d_scan(y_pos)
            
            # Convert distances to z coordinates (surface heights)
            z_2d = -distances
            
            # Filter out NaN points
            valid_mask = ~(np.isnan(x_2d) | np.isnan(y_2d) | np.isnan(distances))
            
            if np.any(valid_mask):
                all_x.extend(x_2d[valid_mask])
                all_y.extend(y_2d[valid_mask])
                all_z.extend(z_2d[valid_mask])
        
        # Convert to numpy arrays
        x_3d = np.array(all_x)
        y_3d = np.array(all_y) 
        z_3d = np.array(all_z)
        
        print(f"3D scan complete. Total points: {len(x_3d)}")
        
        # Apply x-axis rotation if specified
        if self.x_inclination_angle_deg != 0:
            print(f"Applying x-axis rotation: {self.x_inclination_angle_deg:.1f} degrees")
            x_3d, y_3d, z_3d = self.apply_x_rotation(x_3d, y_3d, z_3d)
        
        return x_3d, y_3d, z_3d
    
    def calculate_expected_apex(self) -> Tuple[float, float, float]:
        """
        Calculate the expected apex location based on geometric parameters
        Uses the same coordinate system as the 3D scan (z = -distance from scanner)
        """
        # Get the 2D scanner parameters
        hole_depth = self.scanner_params['hole_depth']
        inner_radius = self.scanner_params['inner_radius']
        csk_angle_rad = np.radians(self.scanner_params['csk_angle'])
        x_offset = self.scanner_params['x_offset']
        standoff = self.scanner_params['standoff']
        
        # Calculate apex position in the 3D scan coordinate system
        # The apex is at the tip of the cone, below the inner radius edge
        # Distance from inner radius to apex along cone axis
        half_angle = csk_angle_rad / 2
        apex_depth_below_inner = inner_radius / np.tan(half_angle)
        outer_rim_depth_variation = self.scanner_params['outer_rim_depth_variation']
        
        # Total depth from surface to apex (includes rim variation)
        total_apex_depth = hole_depth + apex_depth_below_inner + outer_rim_depth_variation
        
        # In the 3D scan coordinate system:
        # - Surface is at z = -standoff (where laser distance = standoff)
        # - Points deeper into material have more negative z values
        # - So apex is at z = -standoff - total_apex_depth
        
        apex_x = x_offset  # Centered at scanner x-offset
        apex_y = 0.0       # Centered (no y-offset for apex calculation)
        apex_z = -standoff - total_apex_depth  # Below surface by total depth
        
        # Apply x-axis rotation if present (same as applied to point cloud)
        if self.x_inclination_angle_rad != 0:
            cos_theta = np.cos(self.x_inclination_angle_rad)
            sin_theta = np.sin(self.x_inclination_angle_rad)
            
            # Apply rotation: y' = cos(θ)*y - sin(θ)*z, z' = sin(θ)*y + cos(θ)*z
            apex_y_rotated = cos_theta * apex_y - sin_theta * apex_z
            apex_z_rotated = sin_theta * apex_y + cos_theta * apex_z
            
            apex_y = apex_y_rotated
            apex_z = apex_z_rotated
        
        return apex_x, apex_y, apex_z
    
    def get_cone_geometry_info(self) -> dict:
        """Get detailed information about the expected cone geometry"""
        hole_depth = self.scanner_params['hole_depth']
        inner_radius = self.scanner_params['inner_radius']
        outer_rim_depth_variation = self.scanner_params['outer_rim_depth_variation']
        csk_angle_rad = np.radians(self.scanner_params['csk_angle'])
        
        half_angle = csk_angle_rad / 2
        apex_depth_below_inner = inner_radius / np.tan(half_angle)
        total_depth = hole_depth + apex_depth_below_inner + outer_rim_depth_variation
        outer_radius = inner_radius + hole_depth / np.tan(half_angle)
        
        apex_x, apex_y, apex_z = self.calculate_expected_apex()
        
        return {
            'apex_location': (apex_x, apex_y, apex_z),
            'total_depth_from_surface': total_depth,
            'hole_depth': hole_depth,
            'apex_depth_below_inner': apex_depth_below_inner,
            'inner_radius': inner_radius,
            'outer_radius': outer_radius,
            'half_angle_deg': np.degrees(half_angle),
            'included_angle_deg': self.scanner_params['csk_angle'],
            'outer_rim_depth_variation': outer_rim_depth_variation
        }
    
    def visualize_point_cloud(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, show_apex: bool = True):
        """Visualize the 3D point cloud with optional expected apex location"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot colored by z-height
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6, label='Scan Points')
        
        # Add expected apex if requested
        if show_apex:
            apex_x, apex_y, apex_z = self.calculate_expected_apex()
            ax.scatter([apex_x], [apex_y], [apex_z], c='red', s=100, marker='*', 
                      edgecolors='black', linewidth=2, label='Expected Apex', zorder=10)
            
            # Get cone geometry info for display
            cone_info = self.get_cone_geometry_info()
            
            # Add text annotation with cone info
            info_text = f"Expected Apex: ({apex_x:.3f}, {apex_y:.3f}, {apex_z:.3f})\n"
            info_text += f"Total Depth: {cone_info['total_depth_from_surface']:.3f} mm\n"
            info_text += f"Cone Angle: {cone_info['included_angle_deg']:.1f}°\n"
            info_text += f"Inner Radius: {cone_info['inner_radius']:.3f} mm"
            
            # Position text in upper corner
            ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                     fontsize=10, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        title = f'3D Countersink Scan Point Cloud ({len(x)} points)'
        if show_apex:
            title += ' with Expected Apex'
        ax.set_title(title)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Surface Height Z (mm)')
        
        # Add legend
        if show_apex:
            ax.legend(loc='upper right')
        
        # Set equal aspect ratio with some padding around apex
        if show_apex:
            apex_x, apex_y, apex_z = self.calculate_expected_apex()
            # Include apex in range calculation
            x_with_apex = np.append(x, apex_x)
            y_with_apex = np.append(y, apex_y)
            z_with_apex = np.append(z, apex_z)
            
            max_range = np.array([x_with_apex.max()-x_with_apex.min(), 
                                y_with_apex.max()-y_with_apex.min(), 
                                z_with_apex.max()-z_with_apex.min()]).max() / 2.0
            mid_x = (x_with_apex.max()+x_with_apex.min()) * 0.5
            mid_y = (y_with_apex.max()+y_with_apex.min()) * 0.5
            mid_z = (z_with_apex.max()+z_with_apex.min()) * 0.5
        else:
            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            mid_z = (z.max()+z.min()) * 0.5
            
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
    
    def save_point_cloud_pcd(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                            filename: str = "countersink_scan.pcd"):
        """Save point cloud as PCD file"""
        print(f"Saving point cloud to {filename}...")
        
        # Create PCD header
        header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(x)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(x)}
DATA ascii
"""
        
        # Write PCD file
        with open(filename, 'w') as f:
            f.write(header)
            for i in range(len(x)):
                f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f}\n")
        
        print(f"Point cloud saved successfully with {len(x)} points")
    
    def save_point_cloud_csv(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            filename: str = "countersink_scan.csv"):
        """Save point cloud as CSV file"""
        print(f"Saving point cloud to {filename}...")
        
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z'])
            for i in range(len(x)):
                writer.writerow([f"{x[i]:.6f}", f"{y[i]:.6f}", f"{z[i]:.6f}"])
        
        print(f"CSV saved successfully with {len(x)} points")
    
    def print_statistics(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Print statistics about the point cloud"""
        print("\n=== 3D Scan Statistics ===")
        print(f"Total points: {len(x)}")
        print(f"X range: {x.min():.3f} to {x.max():.3f} mm (span: {x.max()-x.min():.3f} mm)")
        print(f"Y range: {y.min():.3f} to {y.max():.3f} mm (span: {y.max()-y.min():.3f} mm)")
        print(f"Z range: {z.min():.3f} to {z.max():.3f} mm (span: {z.max()-z.min():.3f} mm)")
        print(f"Y scanning positions: {self.y_steps}")
        print(f"Points per scan: {self.points_per_scan}")
        print(f"X inclination: {self.x_inclination_angle_deg:.1f} degrees")
        
        # Print apex information
        self.print_apex_info()
    
    def print_apex_info(self):
        """Print information about the expected apex location and cone geometry"""
        cone_info = self.get_cone_geometry_info()
        apex_x, apex_y, apex_z = cone_info['apex_location']
        
        print("\n=== Expected Cone Geometry ===")
        print(f"Expected apex location: ({apex_x:.3f}, {apex_y:.3f}, {apex_z:.3f}) mm")
        print(f"Countersink angle: {cone_info['included_angle_deg']:.1f}° (half-angle: {cone_info['half_angle_deg']:.1f}°)")
        print(f"Inner radius: {cone_info['inner_radius']:.3f} mm")
        print(f"Outer radius: {cone_info['outer_radius']:.3f} mm")
        print(f"Hole depth: {cone_info['hole_depth']:.3f} mm")
        print(f"Apex depth below inner edge: {cone_info['apex_depth_below_inner']:.3f} mm")
        print(f"Total depth from surface: {cone_info['total_depth_from_surface']:.3f} mm")
        print(f"Outer rim depth variation: {cone_info['outer_rim_depth_variation']:.3f} mm")


# Example usage
if __name__ == "__main__":
    # Create 3D scanner
    scanner_3d = Countersink3DScanner(
        # 2D scanner parameters
        csk_angle=100.0,
        scanner_range=15.0,
        x_offset=0.0,
        standoff=85.0,
        inner_radius=1.25,
        outer_rim_depth_variation=0.2,
        hole_depth=0.965,
        inclination_angle=0.0,  # 2D scan inclination
        noise_power=0.004,
        dropout_factor=0.02,
        min_angle_deg=-8.0,
        max_angle_deg=8.0,
        # 3D scanning parameters
        y_min=-3.0,
        y_max=3.0,
        y_steps=30,
        points_per_scan=int(2056/2),
        x_inclination_angle_deg=0.0,  # 3D rotation around x-axis
        y_offset_noise=0.02  # mm positioning noise
    )
    
    # Perform 3D scan
    x, y, z = scanner_3d.scan_3d()
    
    # Print statistics
    scanner_3d.print_statistics(x, y, z)
    
    # Save point cloud
    scanner_3d.save_point_cloud_pcd(x, y, z, "countersink_3d_scan.pcd")
    scanner_3d.save_point_cloud_csv(x, y, z, "countersink_3d_scan.csv")
    
    # Visualize point cloud using matplotlib with expected apex
    scanner_3d.visualize_point_cloud(x, y, z, show_apex=True)
    
    # To view with Open3D, use the separate PCD viewer:
    # python pcd_viewer.py countersink_3d_scan.pcd