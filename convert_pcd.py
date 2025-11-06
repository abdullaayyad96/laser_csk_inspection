#!/usr/bin/env python3
"""
PCD Conversion Script
--------------------
Converts PCD files by applying transformations:
- Convert from meters to millimeters (multiply by 1000)
- Flip z-coordinates (multiply by -1)
- Combination of both transformations

The script preserves the original PCD file structure and creates a new file
with the transformed coordinates.

Usage:
    python convert_pcd.py input.pcd output.pcd [--meters-to-mm] [--flip-z]
    python convert_pcd.py input.pcd --auto-suffix [--meters-to-mm] [--flip-z]
    
Examples:
    # Convert from meters to mm and flip z
    python convert_pcd.py data.pcd data_converted.pcd --meters-to-mm --flip-z
    
    # Auto-generate output filename with suffix
    python convert_pcd.py data.pcd --auto-suffix --meters-to-mm --flip-z
    # Creates: data_mm_flipped.pcd
    
Requirements:
    pip install numpy open3d (optional)
"""

import numpy as np
import os
import sys
import argparse
from typing import Tuple, List

class PCDConverter:
    def __init__(self):
        self.verbose = True
    
    def load_pcd_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load PCD file and return coordinates plus header lines
        
        Args:
            filename: Path to PCD file
            
        Returns:
            Tuple of (x, y, z, header_lines)
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        print(f"📂 Loading PCD file: {filename}")
        
        # Method 1: Try Open3D first (faster)
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(filename)
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                raise ValueError("Point cloud is empty")
            
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            # Read header manually for preservation
            header_lines = self._extract_pcd_header(filename)
            
            print(f"✅ Loaded {len(points)} points using Open3D")
            return x, y, z, header_lines
            
        except ImportError:
            print("⚠️  Open3D not available, using manual parsing...")
        except Exception as e:
            print(f"⚠️  Open3D failed: {e}, falling back to manual parsing...")
        
        # Method 2: Manual parsing
        return self._load_pcd_manual(filename)
    
    def _extract_pcd_header(self, filename: str) -> List[str]:
        """Extract header lines from PCD file"""
        header_lines = []
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('DATA'):
                    header_lines.append(line)
                    break
                header_lines.append(line)
        
        return header_lines
    
    def _load_pcd_manual(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load PCD file manually with robust error handling"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header and find data section
        header_lines = []
        data_start = 0
        points_count = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            header_lines.append(line_stripped)
            
            if line_stripped.startswith('POINTS'):
                points_count = int(line_stripped.split()[1])
            elif line_stripped.startswith('DATA'):
                if 'ascii' not in line_stripped.lower():
                    raise ValueError("Only ASCII PCD files are supported")
                data_start = i + 1
                break
        
        if data_start == 0:
            raise ValueError("Could not find DATA section in PCD file")
        
        print(f"📊 PCD Header Info:")
        print(f"   Expected points: {points_count}")
        print(f"   Data starts at line: {data_start + 1}")
        
        # Read point data with filtering
        point_data = []
        invalid_points = 0
        inf_points = 0
        nan_points = 0
        
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if line:
                values = line.split()
                if len(values) >= 3:
                    try:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        
                        # Check for invalid values
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            nan_points += 1
                            continue
                        
                        if np.isinf(x) or np.isinf(y) or np.isinf(z):
                            inf_points += 1
                            continue
                        
                        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                            invalid_points += 1
                            continue
                        
                        point_data.append([x, y, z])
                        
                    except ValueError:
                        invalid_points += 1
                        continue
        
        if len(point_data) == 0:
            raise ValueError("No valid point data found in PCD file")
        
        # Report filtering
        total_filtered = invalid_points + inf_points + nan_points
        if total_filtered > 0:
            print(f"⚠️  Filtered out {total_filtered} invalid points:")
            if inf_points > 0:
                print(f"      - {inf_points} points with infinite values")
            if nan_points > 0:
                print(f"      - {nan_points} points with NaN values")
            if invalid_points > 0:
                print(f"      - {invalid_points} points with other invalid values")
        
        points = np.array(point_data)
        print(f"✅ Loaded {len(points)} valid points (manual parsing)")
        
        return points[:, 0], points[:, 1], points[:, 2], header_lines
    
    def apply_transformations(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                            meters_to_mm: bool = False, flip_z: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply coordinate transformations
        
        Args:
            x, y, z: Original coordinates
            meters_to_mm: Convert from meters to millimeters
            flip_z: Flip z-coordinates
            
        Returns:
            Transformed coordinates
        """
        # Create copies
        x_new = x.copy()
        y_new = y.copy()
        z_new = z.copy()
        
        transformations = []
        
        print(f"\n🔧 Applying transformations:")
        print(f"   Original ranges:")
        print(f"     X: [{np.min(x):.6f}, {np.max(x):.6f}]")
        print(f"     Y: [{np.min(y):.6f}, {np.max(y):.6f}]") 
        print(f"     Z: [{np.min(z):.6f}, {np.max(z):.6f}]")
        
        # Apply meters to mm conversion
        if meters_to_mm:
            x_new *= 1000.0
            y_new *= 1000.0
            z_new *= 1000.0
            transformations.append("meters → mm conversion")
            print(f"   ✅ Applied meters to mm conversion (×1000)")
        
        # Apply z-flip
        if flip_z:
            z_new *= -1.0
            transformations.append("z-axis flip")
            print(f"   ✅ Applied z-axis flip (×-1)")
        
        if not transformations:
            print(f"   ⚠️  No transformations specified")
        else:
            print(f"   Transformed ranges:")
            print(f"     X: [{np.min(x_new):.6f}, {np.max(x_new):.6f}]")
            print(f"     Y: [{np.min(y_new):.6f}, {np.max(y_new):.6f}]")
            print(f"     Z: [{np.min(z_new):.6f}, {np.max(z_new):.6f}]")
        
        return x_new, y_new, z_new
    
    def update_header(self, header_lines: List[str], num_points: int) -> List[str]:
        """Update header to reflect new point count"""
        updated_header = []
        
        for line in header_lines:
            if line.startswith('POINTS'):
                updated_header.append(f"POINTS {num_points}")
            elif line.startswith('WIDTH'):
                updated_header.append(f"WIDTH {num_points}")
            else:
                updated_header.append(line)
        
        return updated_header
    
    def save_pcd_file(self, filename: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                     header_lines: List[str] = None):
        """
        Save transformed coordinates to PCD file
        
        Args:
            filename: Output filename
            x, y, z: Transformed coordinates
            header_lines: Original header lines (optional)
        """
        num_points = len(x)
        
        print(f"💾 Saving transformed PCD: {filename}")
        print(f"   Points to save: {num_points:,}")
        
        # Create or update header
        if header_lines:
            header = self.update_header(header_lines, num_points)
        else:
            # Create minimal header
            header = [
                "# .PCD v0.7 - Point Cloud Data file format",
                "VERSION 0.7",
                "FIELDS x y z",
                "SIZE 4 4 4",
                "TYPE F F F",
                "COUNT 1 1 1",
                f"WIDTH {num_points}",
                "HEIGHT 1",
                "VIEWPOINT 0 0 0 1 0 0 0",
                f"POINTS {num_points}",
                "DATA ascii"
            ]
        
        # Write file
        with open(filename, 'w') as f:
            # Write header
            for line in header:
                f.write(line + '\n')
            
            # Write point data
            for i in range(num_points):
                f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f}\n")
        
        print(f"✅ Successfully saved {num_points:,} points to {filename}")
    
    def convert_pcd(self, input_file: str, output_file: str, 
                   meters_to_mm: bool = False, flip_z: bool = False):
        """
        Convert PCD file with specified transformations
        
        Args:
            input_file: Input PCD filename
            output_file: Output PCD filename 
            meters_to_mm: Convert from meters to millimeters
            flip_z: Flip z-coordinates
        """
        print(f"🚀 Starting PCD conversion")
        print(f"{'='*60}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Transformations: meters_to_mm={meters_to_mm}, flip_z={flip_z}")
        
        try:
            # Load original PCD
            x, y, z, header_lines = self.load_pcd_file(input_file)
            
            # Apply transformations
            x_new, y_new, z_new = self.apply_transformations(x, y, z, meters_to_mm, flip_z)
            
            # Save transformed PCD
            self.save_pcd_file(output_file, x_new, y_new, z_new, header_lines)
            
            print(f"\n✅ Conversion completed successfully!")
            print(f"📁 Output file: {output_file}")
            
            # Show file sizes
            input_size = os.path.getsize(input_file) / 1024 / 1024
            output_size = os.path.getsize(output_file) / 1024 / 1024
            print(f"📊 File sizes: {input_size:.2f} MB → {output_size:.2f} MB")
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            raise

def generate_output_filename(input_file: str, meters_to_mm: bool, flip_z: bool) -> str:
    """Generate output filename with descriptive suffix"""
    base, ext = os.path.splitext(input_file)
    
    suffixes = []
    if meters_to_mm:
        suffixes.append("mm")
    if flip_z:
        suffixes.append("flipped")
    
    if suffixes:
        suffix = "_" + "_".join(suffixes)
    else:
        suffix = "_converted"
    
    return f"{base}{suffix}{ext}"

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Convert PCD files with coordinate transformations')
    parser.add_argument('input_file', help='Input PCD file path')
    parser.add_argument('output_file', nargs='?', help='Output PCD file path')
    parser.add_argument('--auto-suffix', action='store_true', 
                       help='Auto-generate output filename with descriptive suffix')
    parser.add_argument('--meters-to-mm', action='store_true',
                       help='Convert coordinates from meters to millimeters (×1000)')
    parser.add_argument('--flip-z', action='store_true',
                       help='Flip z-coordinates (×-1)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite output file if it exists')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return 1
    
    if not args.input_file.lower().endswith('.pcd'):
        print(f"❌ Input file must be a PCD file: {args.input_file}")
        return 1
    
    # Determine output file
    if args.auto_suffix:
        output_file = generate_output_filename(args.input_file, args.meters_to_mm, args.flip_z)
    elif args.output_file:
        output_file = args.output_file
    else:
        print("❌ Must specify output file or use --auto-suffix")
        return 1
    
    # Check if no transformations specified
    if not args.meters_to_mm and not args.flip_z:
        print("⚠️  Warning: No transformations specified. File will be copied as-is.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return 0
    
    # Check output file exists
    if os.path.exists(output_file) and not args.overwrite:
        print(f"❌ Output file already exists: {output_file}")
        print("Use --overwrite to overwrite existing file")
        return 1
    
    # Create converter and run conversion
    converter = PCDConverter()
    
    try:
        converter.convert_pcd(
            input_file=args.input_file,
            output_file=output_file,
            meters_to_mm=args.meters_to_mm,
            flip_z=args.flip_z
        )
        return 0
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return 1

if __name__ == "__main__":
    exit(main())