#!/usr/bin/env python3
"""
Single PCD Countersink Analysis Script
-------------------------------------
Processes a single PCD file with the complete analysis pipeline:
1. Load PCD file
2. Convert from meters to mm and flip z-axis
3. Split into left/right halves
4. Run countersink depth estimation on each half
5. Generate analysis report

This script reuses existing functionality from:
- test_pcd_loading_and_splitting.py (for PCD loading and splitting)
- convert_pcd.py (for coordinate transformations)
- countersink_depth_estimator.py (for depth estimation)

Usage:
    python single_pcd_analysis.py input.pcd [options]
    
Requirements:
    pip install numpy scipy matplotlib pandas
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Import existing functionality
try:
    from countersink_depth_estimator import CountersinkDepthEstimator
    from test_pcd_loading_and_splitting import PCDTester
    from convert_pcd import PCDConverter
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure all required scripts are in the same directory")
    sys.exit(1)

class SinglePCDAnalyzer:
    def __init__(self, 
                 estimator_params: Dict = None,
                 save_intermediate: bool = True,
                 generate_plots: bool = False):
        """
        Initialize single PCD analyzer
        
        Args:
            estimator_params: Parameters for CountersinkDepthEstimator
            save_intermediate: Save intermediate processed PCD files
            generate_plots: Generate visualization plots for analysis
        """
        self.save_intermediate = save_intermediate
        self.generate_plots = generate_plots
        
        # Initialize components with reusable functionality
        self.pcd_converter = PCDConverter()
        self.pcd_tester = PCDTester(split_method='center', no_use_zone_percent=30)  # Use center split method
        
        # Default estimator parameters
        default_params = {
            'expected_csk_angle_deg': 100.0,
            'expected_inner_radius': 1.2446,
            'optimizer': 'slsqp',        # Match countersink_depth_estimator.py default
            'flip_z': False,  # We handle this in preprocessing
            'meters_to_mm': False,  # We handle this in preprocessing
            'noise_filter': True,
            'noise_neighbors': 100,  # Match countersink_depth_estimator.py default
            'noise_radius': 0.5,     # Match countersink_depth_estimator.py default
            'outlier_method': 'percentile',
            'outlier_threshold': 0.1,
            'random_seed': 42,
            'ransac_sample_points': 3
        }
        
        if estimator_params:
            default_params.update(estimator_params)
        
        self.estimator_params = default_params
        
        print(f"🚀 Single PCD Analyzer initialized")
        print(f"   Estimator parameters: {self.estimator_params}")

    def process_pcd_file(self, pcd_file: str, output_folder: str = None) -> Dict:
        """
        Process a single PCD file and return analysis results
        
        Args:
            pcd_file: Path to the PCD file
            output_folder: Optional output folder for intermediate files
            
        Returns:
            Dictionary containing analysis results
        """
        pcd_path = Path(pcd_file)
        if not pcd_path.exists():
            raise FileNotFoundError(f"PCD file not found: {pcd_file}")
        
        # Set up output folder
        if output_folder is None:
            output_folder = pcd_path.parent / f"{pcd_path.stem}_analysis"
        else:
            output_folder = Path(output_folder)
        
        output_folder.mkdir(exist_ok=True)
        
        print(f"\n🔄 Processing: {pcd_path.name}")
        print(f"   Output folder: {output_folder}")
        
        try:
            # Step 1: Load raw PCD
            print("  📁 Loading raw PCD file...")
            x_raw, y_raw, z_raw, header_lines = self.pcd_converter.load_pcd_file(str(pcd_path))
            print(f"     Loaded {len(x_raw):,} points")
            
            # Step 2: Convert coordinates for entire point cloud first
            print("  🔄 Converting coordinates (meters→mm, flip z) for entire point cloud...")
            x_converted, y_converted, z_converted = self.pcd_converter.apply_transformations(
                x_raw, y_raw, z_raw, meters_to_mm=True, flip_z=True
            )
            print(f"     Converted {len(x_converted):,} points")
            
            # Step 3: Estimate global surface plane from entire point cloud
            print("  📐 Estimating global surface plane from entire point cloud...")
            global_plane_params = self.estimate_global_plane(x_converted, y_converted, z_converted)
            print(f"     Global plane normal: ({global_plane_params['normal'][0]:.3f}, {global_plane_params['normal'][1]:.3f}, {global_plane_params['normal'][2]:.3f})")
            print(f"     Global plane point: ({global_plane_params['point'][0]:.3f}, {global_plane_params['point'][1]:.3f}, {global_plane_params['point'][2]:.3f})")
            
            # Step 4: Split the converted point cloud
            print("  ✂️  Splitting converted point cloud...")
            left_points, right_points, split_line = self.pcd_tester.split_point_cloud(
                x_converted, y_converted, z_converted
            )
            
            # Create split info dictionary
            split_info = {
                'method': self.pcd_tester.split_method,
                'split_line': split_line,
                'no_use_zone_percent': self.pcd_tester.no_use_zone_percent,
                'left_count': len(left_points['x']),
                'right_count': len(right_points['x']),
                'global_plane_params': global_plane_params
            }
            
            print(f"     Left points: {len(left_points['x']):,}")
            print(f"     Right points: {len(right_points['x']):,}")
            print(f"     Split line: {split_line:.3f}")
            
            # Convert split points to the format expected by the estimator
            left_converted = left_points  # Already in dictionary format
            right_converted = right_points  # Already in dictionary format
            
            print(f"     Left converted: {len(left_converted['x']):,} points")
            print(f"     Right converted: {len(right_converted['x']):,} points")
            
            # Step 4: Save intermediate files if requested
            processed_files = {}
            if self.save_intermediate:
                print("  💾 Saving processed halves...")
                processed_files = self.save_split_pcds(
                    pcd_path, left_converted, right_converted, output_folder
                )
            
            # Step 5: Run depth estimation on each half using global plane parameters
            print("  🎯 Running depth estimation with global plane parameters...")
            # Estimate right side
            right_results = self.estimate_depth_from_arrays(
                right_converted, 'right', global_plane_params
            )

            # Estimate left side
            left_results = self.estimate_depth_from_arrays(
                left_converted, 'left', global_plane_params
            )
            
            # Step 6: Print depth results
            self.print_depth_results(left_results, right_results)
            
            # Step 7: Compile results (simplified)
            analysis_results = {
                'input_file': str(pcd_path),
                'output_folder': str(output_folder),
                'timestamp': datetime.now().isoformat(),
                'original_points': len(x_raw),
                'split_info': split_info,
                'processed_files': processed_files,
                'left_results': left_results,
                'right_results': right_results,
                'estimator_params': self.estimator_params
            }
            
            # Step 8: Save results
            self.save_analysis_results(analysis_results, output_folder)
            
            # Step 9: Generate plots if requested
            if self.generate_plots:
                self.generate_analysis_plots(analysis_results, output_folder)
            
            print(f"  ✅ Analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            raise

    def estimate_global_plane(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
        """
        Estimate surface plane parameters from entire point cloud
        
        Args:
            x, y, z: Point cloud coordinates (already converted)
            
        Returns:
            Dictionary with global plane parameters
        """
        # Create a temporary estimator to get plane parameters
        temp_estimator = CountersinkDepthEstimator(**self.estimator_params)
        
        # Apply noise filtering
        x_filtered, y_filtered, z_filtered, noise_mask = temp_estimator.filter_noise(x, y, z)
        
        # Estimate surface plane using the same method as the estimator
        surface_normal, surface_point, surface_threshold = temp_estimator.estimate_surface_plane(
            x_filtered, y_filtered, z_filtered
        )
        
        return {
            'normal': surface_normal,
            'point': surface_point,
            'threshold': surface_threshold,
            'filtered_points': len(x_filtered),
            'original_points': len(x)
        }

    def estimate_depth_from_arrays(self, points: Dict, side: str, global_plane_params: Dict = None) -> Dict:
        """
        Estimate countersink depth from point arrays
        
        Args:
            points: Dictionary with 'x', 'y', 'z' arrays
            side: 'left' or 'right'
            global_plane_params: Global plane parameters to use instead of local estimation
            
        Returns:
            Dictionary with estimation results
        """
        print(f"    🎯 Estimating depth for {side} side...")
        
        # Create estimator with configured parameters
        estimator = CountersinkDepthEstimator(**self.estimator_params)
        
        try:
            if global_plane_params is not None:
                print(f"       🌐 Using global plane parameters for {side} side")
                # Use the estimator's new global plane method directly
                results = estimator.estimate_depth_with_global_plane(
                    points['x'], points['y'], points['z'], global_plane_params
                )
            else:
                # Use the standard estimation method
                results = estimator.estimate_depth_from_arrays(
                    points['x'], points['y'], points['z']
                )
            
            # Add side identification
            results['side'] = side
            results['success'] = True
            results['error_message'] = None
            
            print(f"       ✅ {side.capitalize()} depth: {results['estimated_total_depth']:.6f} mm")
            print(f"       RMSE: {results['fit_rmse']:.6f}")
            
            return results
            
        except Exception as e:
            print(f"       ❌ {side.capitalize()} estimation failed: {e}")
            
            # Return failure result
            return {
                'side': side,
                'success': False,
                'error_message': str(e),
                'estimated_total_depth': np.nan,
                'estimated_apex_depth': np.nan,
                'fit_rmse': np.nan,
                'total_points': len(points['x']) if 'x' in points else 0,
                'surface_points': 0,
                'hole_points': 0
            }

    def print_depth_results(self, left_results: Dict, right_results: Dict):
        """Print depth results in a clear, visual format"""
        print("\n" + "="*60)
        print("🎯 COUNTERSINK DEPTH ANALYSIS RESULTS")
        print("="*60)
        
        # Left side results
        print(f"\n📍 LEFT SIDE:")
        if left_results['success']:
            print(f"   ✅ Depth: {left_results['estimated_total_depth']:.3f} mm")
            print(f"   📊 RMSE: {left_results['fit_rmse']:.3f} mm")
            print(f"   🔢 Points: {left_results['hole_points']:,} hole points")
        else:
            print(f"   ❌ Failed: {left_results.get('error_message', 'Unknown error')}")
        
        # Right side results  
        print(f"\n📍 RIGHT SIDE:")
        if right_results['success']:
            print(f"   ✅ Depth: {right_results['estimated_total_depth']:.3f} mm")
            print(f"   📊 RMSE: {right_results['fit_rmse']:.3f} mm")
            print(f"   🔢 Points: {right_results['hole_points']:,} hole points")
        else:
            print(f"   ❌ Failed: {right_results.get('error_message', 'Unknown error')}")
        
        # Summary if both successful
        if left_results['success'] and right_results['success']:
            avg_depth = (left_results['estimated_total_depth'] + right_results['estimated_total_depth']) / 2
            print(f"\n📈 SUMMARY:")
            print(f"   Average depth: {avg_depth:.3f} mm")
            print(f"   Left:  {left_results['estimated_total_depth']:.3f} mm")
            print(f"   Right: {right_results['estimated_total_depth']:.3f} mm")
        
        print("="*60)

    def analyze_bilateral_consistency(self, left_results: Dict, right_results: Dict) -> Dict:
        """Analyze consistency between left and right estimations"""
        analysis = {
            'both_successful': left_results['success'] and right_results['success'],
            'left_successful': left_results['success'],
            'right_successful': right_results['success']
        }
        
        if analysis['both_successful']:
            # Calculate differences
            depth_diff = abs(left_results['estimated_total_depth'] - right_results['estimated_total_depth'])
            apex_diff = abs(left_results['estimated_apex_depth'] - right_results['estimated_apex_depth'])
            
            # Calculate averages
            avg_depth = (left_results['estimated_total_depth'] + right_results['estimated_total_depth']) / 2
            avg_apex = (left_results['estimated_apex_depth'] + right_results['estimated_apex_depth']) / 2
            
            # Relative differences
            relative_depth_diff = (depth_diff / avg_depth) * 100 if avg_depth > 0 else np.inf
            relative_apex_diff = (apex_diff / avg_apex) * 100 if avg_apex > 0 else np.inf
            
            # Quality assessment
            analysis.update({
                'depth_difference_mm': depth_diff,
                'apex_difference_mm': apex_diff,
                'average_depth_mm': avg_depth,
                'average_apex_depth_mm': avg_apex,
                'relative_depth_difference_percent': relative_depth_diff,
                'relative_apex_difference_percent': relative_apex_diff,
                'depth_consistency': 'excellent' if relative_depth_diff < 1 else
                                   'good' if relative_depth_diff < 5 else
                                   'fair' if relative_depth_diff < 10 else 'poor',
                'left_rmse': left_results['fit_rmse'],
                'right_rmse': right_results['fit_rmse'],
                'average_rmse': (left_results['fit_rmse'] + right_results['fit_rmse']) / 2
            })
        else:
            # Handle cases where one or both sides failed
            analysis.update({
                'depth_difference_mm': np.nan,
                'apex_difference_mm': np.nan,
                'average_depth_mm': np.nan,
                'average_apex_depth_mm': np.nan,
                'relative_depth_difference_percent': np.nan,
                'relative_apex_difference_percent': np.nan,
                'depth_consistency': 'failed',
                'left_rmse': left_results.get('fit_rmse', np.nan),
                'right_rmse': right_results.get('fit_rmse', np.nan),
                'average_rmse': np.nan
            })
        
        return analysis

    def save_split_pcds(self, original_file: Path, left_points: Dict, right_points: Dict, 
                       output_folder: Path) -> Dict:
        """Save the split and processed PCD halves"""
        processed_files = {}
        
        # Generate filenames
        base_name = original_file.stem
        left_filename = output_folder / f"{base_name}_left_processed.pcd"
        right_filename = output_folder / f"{base_name}_right_processed.pcd"
        
        # Save left half
        self.pcd_tester.save_pcd_half(left_points, str(left_filename), "left")
        processed_files['left'] = str(left_filename)
        
        # Save right half
        self.pcd_tester.save_pcd_half(right_points, str(right_filename), "right")
        processed_files['right'] = str(right_filename)
        
        print(f"     Saved: {left_filename.name}")
        print(f"     Saved: {right_filename.name}")
        
        return processed_files

    def save_analysis_results(self, results: Dict, output_folder: Path):
        """Save analysis results to JSON and summary text files"""
        
        # Save complete results as JSON
        json_file = output_folder / "analysis_results.json"
        with open(json_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_numpy_types(results)
            json.dump(json_results, f, indent=2)
        
        # Save human-readable summary
        summary_file = output_folder / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            self._write_summary_report(results, f)
        
        print(f"     Saved: {json_file.name}")
        print(f"     Saved: {summary_file.name}")

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _write_summary_report(self, results: Dict, file_handle):
        """Write human-readable summary report"""
        file_handle.write("COUNTERSINK DEPTH ANALYSIS SUMMARY\n")
        file_handle.write("=" * 50 + "\n\n")
        
        file_handle.write(f"Input File: {results['input_file']}\n")
        file_handle.write(f"Analysis Date: {results['timestamp']}\n")
        file_handle.write(f"Original Points: {results['original_points']:,}\n\n")
        
        # Split information
        file_handle.write("SPLIT INFORMATION\n")
        file_handle.write("-" * 20 + "\n")
        split_info = results['split_info']
        file_handle.write(f"Split method: {split_info.get('method', 'unknown')}\n")
        file_handle.write(f"Split line: {split_info.get('split_line', 'unknown'):.3f}\n")
        file_handle.write(f"No-use zone: {split_info.get('no_use_zone_percent', 0)}%\n\n")
        
        # Left results
        file_handle.write("LEFT SIDE ANALYSIS\n")
        file_handle.write("-" * 20 + "\n")
        left = results['left_results']
        if left['success']:
            file_handle.write(f"Total Depth: {left['estimated_total_depth']:.6f} mm\n")
            file_handle.write(f"Apex Depth: {left['estimated_apex_depth']:.6f} mm\n")
            file_handle.write(f"RMSE: {left['fit_rmse']:.6f} mm\n")
            file_handle.write(f"Surface Points: {left['surface_points']:,}\n")
            file_handle.write(f"Hole Points: {left['hole_points']:,}\n")
        else:
            file_handle.write(f"FAILED: {left['error_message']}\n")
        file_handle.write("\n")
        
        # Right results
        file_handle.write("RIGHT SIDE ANALYSIS\n")
        file_handle.write("-" * 20 + "\n")
        right = results['right_results']
        if right['success']:
            file_handle.write(f"Total Depth: {right['estimated_total_depth']:.6f} mm\n")
            file_handle.write(f"Apex Depth: {right['estimated_apex_depth']:.6f} mm\n")
            file_handle.write(f"RMSE: {right['fit_rmse']:.6f} mm\n")
            file_handle.write(f"Surface Points: {right['surface_points']:,}\n")
            file_handle.write(f"Hole Points: {right['hole_points']:,}\n")
        else:
            file_handle.write(f"FAILED: {right['error_message']}\n")
        file_handle.write("\n")
        
        # Summary if both successful
        if left['success'] and right['success']:
            avg_depth = (left['estimated_total_depth'] + right['estimated_total_depth']) / 2
            depth_diff = abs(left['estimated_total_depth'] - right['estimated_total_depth'])
            
            file_handle.write("ANALYSIS SUMMARY\n")
            file_handle.write("-" * 20 + "\n")
            file_handle.write(f"Average Depth: {avg_depth:.6f} mm\n")
            file_handle.write(f"Depth Difference: {depth_diff:.6f} mm\n")
            file_handle.write(f"Left Depth: {left['estimated_total_depth']:.6f} mm\n")
            file_handle.write(f"Right Depth: {right['estimated_total_depth']:.6f} mm\n")
        else:
            file_handle.write("ANALYSIS SUMMARY\n")
            file_handle.write("-" * 20 + "\n")
            file_handle.write("Analysis incomplete - one or both sides failed\n")

    def generate_analysis_plots(self, results: Dict, output_folder: Path):
        """Generate three separate visualization figures: clustering, right hole, and left hole"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            print(f"     📊 Generating 3D visualization plots...")
            
            left_results = results['left_results']
            right_results = results['right_results']
            
            # Get the original point cloud data for visualization
            pcd_path = Path(results['input_file'])
            x_raw, y_raw, z_raw, _ = self.pcd_converter.load_pcd_file(str(pcd_path))
            
            # Apply the same transformations used in processing (entire cloud first)
            x_converted, y_converted, z_converted = self.pcd_converter.apply_transformations(
                x_raw, y_raw, z_raw, meters_to_mm=True, flip_z=True
            )
            
            # Get global plane parameters for segmentation visualization
            global_plane_params = results['split_info']['global_plane_params']
            
            # Segment the entire point cloud using global plane parameters
            points_all = np.column_stack([x_converted, y_converted, z_converted])
            surface_normal = global_plane_params['normal']
            surface_point = global_plane_params['point']
            surface_threshold = global_plane_params['threshold']
            
            distances_to_plane = np.abs(np.dot(points_all - surface_point, surface_normal))
            surface_mask_global = distances_to_plane <= surface_threshold
            hole_mask_global = distances_to_plane > surface_threshold
            
            # Split the converted point cloud for visualization
            left_points, right_points, split_line = self.pcd_tester.split_point_cloud(
                x_converted, y_converted, z_converted
            )
            
            # The points are already in the right format (dictionaries)
            left_converted = left_points
            right_converted = right_points
            
            # === FIGURE 1: CLUSTERING/SEGMENTATION ANALYSIS ===
            print("     🌐 Creating Figure 1: Clustering/Segmentation Analysis...")
            self.create_clustering_figure(results, output_folder, x_converted, y_converted, z_converted,
                                        global_plane_params, points_all, surface_mask_global, hole_mask_global,
                                        left_converted, right_converted, split_line)
            
            # === FIGURE 2: RIGHT HOLE ANALYSIS ===
            if right_results['success']:
                print("     🟠 Creating Figure 2: Right Hole Analysis...")
                self.create_right_hole_figure(results, output_folder, right_converted, right_results)
            
            # === FIGURE 3: LEFT HOLE ANALYSIS ===
            if left_results['success']:
                print("     🔵 Creating Figure 3: Left Hole Analysis...")
                self.create_left_hole_figure(results, output_folder, left_converted, left_results)
            
            print(f"     ✅ All visualization figures generated successfully!")
            
        except Exception as e:
            print(f"     ❌ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

    def create_clustering_figure(self, results: Dict, output_folder: Path, 
                               x_converted, y_converted, z_converted, global_plane_params,
                               points_all, surface_mask_global, hole_mask_global,
                               left_converted, right_converted, split_line):
        """Create Figure 1: Clustering/Segmentation Analysis"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig1 = plt.figure(figsize=(16, 12))
        fig1.suptitle(f"Figure 1: Clustering and Segmentation Analysis - {Path(results['input_file']).name}", 
                     fontsize=14, fontweight='bold')
        
        surface_normal = global_plane_params['normal']
        surface_point = global_plane_params['point']
        surface_threshold = global_plane_params['threshold']
        surface_points_global = points_all[surface_mask_global]
        hole_points_global = points_all[hole_mask_global]
        
        # 1. Overall original point cloud (2x2 grid)
        ax1 = fig1.add_subplot(2, 2, 1, projection='3d')
        scatter = ax1.scatter(x_converted, y_converted, z_converted, 
                             c=z_converted, cmap='viridis', s=0.5, alpha=0.6)
        ax1.set_title(f'Overall Point Cloud\n({len(x_converted):,} points)')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        plt.colorbar(scatter, ax=ax1, shrink=0.6)
        
        # 2. Global plane/hole segmentation
        ax2 = fig1.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(surface_points_global[:, 0], surface_points_global[:, 1], surface_points_global[:, 2],
                   c='lightblue', s=0.5, alpha=0.7, label=f'Surface ({len(surface_points_global):,})')
        ax2.scatter(hole_points_global[:, 0], hole_points_global[:, 1], hole_points_global[:, 2],
                   c='red', s=1.5, alpha=0.9, label=f'Hole ({len(hole_points_global):,})')
        
        # Plot global plane
        self.plot_plane_surface(ax2, surface_normal, surface_point, size=15)
        
        ax2.set_title('Global Plane/Hole Segmentation')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        ax2.legend()
        
        # 3. Split visualization with segmentation
        ax3 = fig1.add_subplot(2, 2, 3, projection='3d')
        # Color left points blue, right points orange, but use intensity for surface/hole
        left_mask = np.array(x_converted) < split_line
        right_mask = np.array(x_converted) >= split_line
        
        # Left surface and hole
        left_surface_mask = left_mask & surface_mask_global
        left_hole_mask = left_mask & hole_mask_global
        # Right surface and hole  
        right_surface_mask = right_mask & surface_mask_global
        right_hole_mask = right_mask & hole_mask_global
        
        ax3.scatter(x_converted[left_surface_mask], y_converted[left_surface_mask], z_converted[left_surface_mask],
                   c='lightblue', s=0.5, alpha=0.6, label=f'Left Surface ({np.sum(left_surface_mask):,})')
        ax3.scatter(x_converted[left_hole_mask], y_converted[left_hole_mask], z_converted[left_hole_mask],
                   c='blue', s=1.5, alpha=0.8, label=f'Left Hole ({np.sum(left_hole_mask):,})')
        ax3.scatter(x_converted[right_surface_mask], y_converted[right_surface_mask], z_converted[right_surface_mask],
                   c='lightyellow', s=0.5, alpha=0.6, label=f'Right Surface ({np.sum(right_surface_mask):,})')
        ax3.scatter(x_converted[right_hole_mask], y_converted[right_hole_mask], z_converted[right_hole_mask],
                   c='orange', s=1.5, alpha=0.8, label=f'Right Hole ({np.sum(right_hole_mask):,})')
        
        # Draw split line
        z_range = [np.min(z_converted), np.max(z_converted)]
        y_range = [np.min(y_converted), np.max(y_converted)]
        ax3.plot([split_line, split_line], y_range, z_range, 'k--', linewidth=2, alpha=0.8, label='Split Line')
        
        ax3.set_title(f'Split View with Segmentation\nSplit at X = {split_line:.3f} mm')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        ax3.legend(fontsize=8)
        
        # 4. Global segmentation statistics
        ax4 = fig1.add_subplot(2, 2, 4)
        categories = ['Surface\nPoints', 'Hole\nPoints']
        counts = [len(surface_points_global), len(hole_points_global)]
        colors = ['lightblue', 'red']
        
        bars = ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Point Count')
        ax4.set_title('Global Segmentation Summary')
        ax4.grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Add percentage labels
        total_points = sum(counts)
        percentages = [count/total_points*100 for count in counts]
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                    f'{pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        
        ax4.text(0.5, 0.95, f'Total: {total_points:,} points\nThreshold: {surface_threshold:.3f} mm',
                transform=ax4.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        
        # Save Figure 1
        plot_file1 = output_folder / "figure1_clustering_segmentation.png"
        plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
        print(f"     📊 Saved: {plot_file1.name}")
        plt.show()

    def create_right_hole_figure(self, results: Dict, output_folder: Path, right_converted, right_results):
        """Create Figure 2: Right Hole Analysis - matching countersink_depth_estimator.py style"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig2 = plt.figure(figsize=(20, 12))
        fig2.suptitle(f"Figure 2: Right Hole Analysis - {Path(results['input_file']).name}", 
                     fontsize=16, fontweight='bold')
        
        # Create estimator and run analysis to get detailed results
        estimator = CountersinkDepthEstimator(**self.estimator_params)
        _ = estimator.estimate_depth_from_arrays(right_converted['x'], right_converted['y'], right_converted['z'])
        
        # Apply same preprocessing as in original estimator
        x_proc, y_proc, z_proc = estimator.preprocess_point_cloud(
            right_converted['x'], right_converted['y'], right_converted['z']
        )
        x_filtered, y_filtered, z_filtered, noise_mask = estimator.filter_noise(
            x_proc, y_proc, z_proc
        )
        
        # Get segmentation using estimator's method
        if hasattr(estimator, 'surface_points') and hasattr(estimator, 'hole_points'):
            surface_points = estimator.surface_points
            hole_points = estimator.hole_points
        else:
            # Fallback segmentation
            surface_points, hole_points = estimator.segment_surface_and_hole(
                x_filtered, y_filtered, z_filtered
            )
        
        # 1. Original point cloud with noise filtering visualization (EXACTLY like original)
        ax1 = fig2.add_subplot(2, 3, 1, projection='3d')
        if estimator.noise_filter:
            # Show noise points and filtered points
            noise_points_mask = ~noise_mask
            if np.any(noise_points_mask):
                ax1.scatter(x_proc[noise_points_mask], y_proc[noise_points_mask], z_proc[noise_points_mask], 
                           c='red', s=3, alpha=0.8, label=f'Noise ({np.sum(noise_points_mask)})')
            ax1.scatter(x_filtered, y_filtered, z_filtered, c=z_filtered, cmap='viridis', s=1, alpha=0.6, label='Filtered')
            ax1.set_title('Right: Noise Filtering Results')
        else:
            ax1.scatter(x_filtered, y_filtered, z_filtered, c=z_filtered, cmap='viridis', s=1, alpha=0.6)
            ax1.set_title('Right: Original Point Cloud')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.legend()
        
        # 2. Segmented point cloud (EXACTLY like original estimator)
        ax2 = fig2.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(surface_points['x'], surface_points['y'], surface_points['z'], 
                   c='blue', s=1, alpha=0.6, label='Surface')
        ax2.scatter(hole_points['x'], hole_points['y'], hole_points['z'], 
                   c='red', s=1, alpha=0.8, label='Hole')
        ax2.set_title('Right: Segmented Points')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        ax2.legend()
        
        # 3. Fitted cone with outlier visualization (EXACTLY like original estimator)
        ax3 = fig2.add_subplot(2, 3, 3, projection='3d')
        
        if 'cone_fit' in estimator.results:
            cone_fit = estimator.results['cone_fit']
            
            # Plot inliers and outliers exactly like original estimator
            if cone_fit.get('outlier_coords') is not None and cone_fit.get('inlier_coords') is not None:
                outlier_coords = cone_fit['outlier_coords']
                inlier_coords = cone_fit['inlier_coords']
                
                print(f"Debug: Right inlier coords count: {len(inlier_coords['x'])}")
                print(f"Debug: Right outlier coords count: {len(outlier_coords['x'])}")
                print(f"Debug: Right total hole points: {len(hole_points['x'])}")
                
                # Plot inliers in green (hole points that fit the cone well)
                if len(inlier_coords['x']) > 0:
                    ax3.scatter(inlier_coords['x'], inlier_coords['y'], inlier_coords['z'], 
                               c='green', s=2, alpha=0.8, label=f'Hole Inliers ({len(inlier_coords["x"])})')
                
                # Plot outliers in red (hole points that don't fit the cone well)
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
            
            # Use the estimator's own cone plotting function
            estimator.plot_fitted_cone(ax3, apex, axis, half_angle, cone_height=2.0)
            
            ax3.set_title('Right: Cone Fit - Hole Points Only (Inliers vs Outliers)')
            ax3.legend()
        else:
            ax3.scatter(hole_points['x'], hole_points['y'], hole_points['z'], 
                       c='red', s=1, alpha=0.6, label='Hole points')
            ax3.set_title('Right: Hole Points (No Cone Fit Available)')
        
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        
        # 4. Outlier analysis histogram (like original estimator)
        ax4 = fig2.add_subplot(2, 3, 4)
        if 'cone_fit' in estimator.results and estimator.results['cone_fit'].get('outlier_coords') is not None:
            cone_fit = estimator.results['cone_fit']
            outlier_coords = cone_fit['outlier_coords']
            inlier_coords = cone_fit['inlier_coords']
            
            if len(outlier_coords['x']) > 0 and len(inlier_coords['x']) > 0:
                # Plot Z-distribution comparison
                ax4.hist(inlier_coords['z'], bins=30, alpha=0.6, label='Inliers', color='green', density=True)
                ax4.hist(outlier_coords['z'], bins=30, alpha=0.6, label='Outliers', color='red', density=True)
                ax4.set_xlabel('Z coordinate (mm)')
                ax4.set_ylabel('Density')
                ax4.set_title('Right: Z-Distribution (Inliers vs Outliers)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # 5. Residual analysis placeholder
        ax5 = fig2.add_subplot(2, 3, 5)
        if 'cone_fit' in estimator.results:
            cone_fit = estimator.results['cone_fit']
            ax5.text(0.5, 0.5, 'Right: Residual Analysis\n(To be implemented)', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Right: Residual Analysis')
        
        # 6. Summary statistics (like original estimator)
        ax6 = fig2.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = "RIGHT SIDE ANALYSIS\n"
        summary_text += "="*40 + "\n"
        summary_text += f"Original points: {len(right_converted['x']):,}\n"
        
        if estimator.noise_filter:
            noise_removed = len(x_proc) - len(x_filtered)
            summary_text += f"Noise points removed: {noise_removed:,}\n"
        
        summary_text += f"Surface points: {len(surface_points['x']):,}\n"
        summary_text += f"Hole points: {len(hole_points['x']):,}\n\n"
        
        if 'cone_fit' in estimator.results:
            cone_fit = estimator.results['cone_fit']
            summary_text += "CONE FITTING:\n"
            summary_text += f"Half-angle: {cone_fit['half_angle_deg']:.1f}°\n"
            summary_text += f"RMSE: {cone_fit['rmse']:.4f} mm\n"
            summary_text += f"Outliers removed: {cone_fit.get('n_outliers_removed', 0)}\n"
            summary_text += f"Inliers used: {cone_fit.get('n_inliers_used', 0)}\n\n"
        
        if right_results['success']:
            summary_text += "DEPTH ESTIMATION:\n"
            summary_text += f"Total depth: {right_results['estimated_total_depth']:.3f} mm\n"
            summary_text += f"Apex depth: {right_results['estimated_apex_depth']:.3f} mm\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the figure
        plot_file = output_folder / "right_hole_detailed_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"     📊 Saved: {plot_file.name}")
        
        # Show the plot
        plt.show()
        
        # 3. Right fitted cone with inliers/outliers
        ax3 = fig2.add_subplot(2, 3, 3, projection='3d')
        if hasattr(estimator, 'results') and 'cone_fit' in estimator.results:
            cone_fit = estimator.results['cone_fit']
            
            # Plot inliers and outliers with clear distinction
            if cone_fit.get('outlier_coords') is not None and cone_fit.get('inlier_coords') is not None:
                inlier_coords = cone_fit['inlier_coords']
                outlier_coords = cone_fit['outlier_coords']
                
                if len(inlier_coords['x']) > 0:
                    ax3.scatter(inlier_coords['x'], inlier_coords['y'], inlier_coords['z'], 
                               c='green', s=2.5, alpha=0.8, label=f'Inliers ({len(inlier_coords["x"]):,})')
                
                if len(outlier_coords['x']) > 0:
                    ax3.scatter(outlier_coords['x'], outlier_coords['y'], outlier_coords['z'], 
                               c='red', s=4, alpha=0.9, label=f'Outliers ({len(outlier_coords["x"]):,})')
            
            # Plot fitted cone
            apex = cone_fit['apex']
            axis = cone_fit['axis']
            half_angle = cone_fit['half_angle_rad']
            
            self.plot_fitted_cone(ax3, apex, axis, half_angle, cone_height=2.0)
            
            ax3.legend(fontsize=9)
        else:
            ax3.scatter(right_converted['x'], right_converted['y'], right_converted['z'], 
                       c='gray', s=1, alpha=0.6)
        
        ax3.set_title(f'Right Side - Fitted Cone\nDepth: {right_results["estimated_total_depth"]:.3f} mm')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        
        # 4. Right segmentation statistics
        ax4 = fig2.add_subplot(2, 3, 4)
        if hasattr(estimator, 'surface_points') and hasattr(estimator, 'hole_points'):
            categories = ['Surface', 'Hole']
            counts = [len(estimator.surface_points['x']), len(estimator.hole_points['x'])]
            colors = ['lightyellow', 'darkorange']
            
            bars = ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
            ax4.set_ylabel('Point Count')
            ax4.set_title('Right Side - Point Distribution')
            ax4.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            # Add RMSE info
            ax4.text(0.5, 0.85, f'RMSE: {right_results["fit_rmse"]:.3f} mm\nDepth: {right_results["estimated_total_depth"]:.3f} mm',
                    transform=ax4.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        # 5. Right depth profile/histogram
        ax5 = fig2.add_subplot(2, 3, 5)
        if hasattr(estimator, 'hole_points') and len(estimator.hole_points['x']) > 0:
            ax5.hist(estimator.hole_points['z'], bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax5.axvline(right_results['estimated_total_depth'], color='red', linestyle='--', 
                       label=f'Depth: {right_results["estimated_total_depth"]:.3f} mm')
            ax5.set_xlabel('Z coordinate (mm)')
            ax5.set_ylabel('Point count')
            ax5.set_title('Right Side - Depth Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Right results summary
        ax6 = fig2.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""Right Side Results:

Depth: {right_results['estimated_total_depth']:.3f} mm
Apex Depth: {right_results['estimated_apex_depth']:.3f} mm
RMSE: {right_results['fit_rmse']:.3f} mm

Point Counts:
Total: {right_results['total_points']:,}
Surface: {right_results['surface_points']:,}
Hole: {right_results['hole_points']:,}

Cone Parameters:
Apex: ({right_results['cone_apex'][0]:.1f}, {right_results['cone_apex'][1]:.1f}, {right_results['cone_apex'][2]:.1f})
Half-angle: {np.degrees(right_results['cone_half_angle']):.1f}°
Optimization Cost: {right_results.get('optimization_cost', 'N/A')}"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax6.set_title('Right Side Summary', fontweight='bold')
        
        plt.tight_layout()
        
        # Save Figure 2
        plot_file2 = output_folder / "figure2_right_hole_analysis.png"
        plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
        print(f"     📊 Saved: {plot_file2.name}")
        plt.show()

    def create_left_hole_figure(self, results: Dict, output_folder: Path, left_converted, left_results):
        """Create Figure 3: Left Hole Analysis - matching countersink_depth_estimator.py style"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig3 = plt.figure(figsize=(20, 12))
        fig3.suptitle(f"Figure 3: Left Hole Analysis - {Path(results['input_file']).name}", 
                     fontsize=16, fontweight='bold')
        
        # Create estimator and run analysis to get detailed results
        estimator = CountersinkDepthEstimator(**self.estimator_params)
        _ = estimator.estimate_depth_from_arrays(left_converted['x'], left_converted['y'], left_converted['z'])
        
        # Apply same preprocessing as in original estimator
        x_proc, y_proc, z_proc = estimator.preprocess_point_cloud(
            left_converted['x'], left_converted['y'], left_converted['z']
        )
        x_filtered, y_filtered, z_filtered, noise_mask = estimator.filter_noise(
            x_proc, y_proc, z_proc
        )
        
        # Get segmentation using estimator's method
        if hasattr(estimator, 'surface_points') and hasattr(estimator, 'hole_points'):
            surface_points = estimator.surface_points
            hole_points = estimator.hole_points
        else:
            # Fallback segmentation
            surface_points, hole_points = estimator.segment_surface_and_hole(
                x_filtered, y_filtered, z_filtered
            )
        
        # 1. Original point cloud with noise filtering visualization (EXACTLY like original)
        ax1 = fig3.add_subplot(2, 3, 1, projection='3d')
        if estimator.noise_filter:
            # Show noise points and filtered points
            noise_points_mask = ~noise_mask
            if np.any(noise_points_mask):
                ax1.scatter(x_proc[noise_points_mask], y_proc[noise_points_mask], z_proc[noise_points_mask], 
                           c='red', s=3, alpha=0.8, label=f'Noise ({np.sum(noise_points_mask)})')
            ax1.scatter(x_filtered, y_filtered, z_filtered, c=z_filtered, cmap='viridis', s=1, alpha=0.6, label='Filtered')
            ax1.set_title('Left: Noise Filtering Results')
        else:
            ax1.scatter(x_filtered, y_filtered, z_filtered, c=z_filtered, cmap='viridis', s=1, alpha=0.6)
            ax1.set_title('Left: Original Point Cloud')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.legend()
        
        # 2. Segmented point cloud (EXACTLY like original estimator)
        ax2 = fig3.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(surface_points['x'], surface_points['y'], surface_points['z'], 
                   c='blue', s=1, alpha=0.6, label='Surface')
        ax2.scatter(hole_points['x'], hole_points['y'], hole_points['z'], 
                   c='red', s=1, alpha=0.8, label='Hole')
        ax2.set_title('Left: Segmented Points')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        ax2.legend()
        
        # 3. Fitted cone with outlier visualization (EXACTLY like original estimator)
        ax3 = fig3.add_subplot(2, 3, 3, projection='3d')
        
        if 'cone_fit' in estimator.results:
            cone_fit = estimator.results['cone_fit']
            
            # Plot inliers and outliers exactly like original estimator
            if cone_fit.get('outlier_coords') is not None and cone_fit.get('inlier_coords') is not None:
                outlier_coords = cone_fit['outlier_coords']
                inlier_coords = cone_fit['inlier_coords']
                
                print(f"Debug: Left inlier coords count: {len(inlier_coords['x'])}")
                print(f"Debug: Left outlier coords count: {len(outlier_coords['x'])}")
                print(f"Debug: Left total hole points: {len(hole_points['x'])}")
                
                # Plot inliers in green (hole points that fit the cone well)
                if len(inlier_coords['x']) > 0:
                    ax3.scatter(inlier_coords['x'], inlier_coords['y'], inlier_coords['z'], 
                               c='green', s=2, alpha=0.8, label=f'Hole Inliers ({len(inlier_coords["x"])})')
                
                # Plot outliers in red (hole points that don't fit the cone well)
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
            
            # Use the estimator's own cone plotting function
            estimator.plot_fitted_cone(ax3, apex, axis, half_angle, cone_height=2.0)
            
            ax3.set_title('Left: Cone Fit - Hole Points Only (Inliers vs Outliers)')
            ax3.legend()
        else:
            ax3.scatter(hole_points['x'], hole_points['y'], hole_points['z'], 
                       c='red', s=1, alpha=0.6, label='Hole points')
            ax3.set_title('Left: Hole Points (No Cone Fit Available)')
        
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        
        # 4. Outlier analysis histogram (like original estimator)
        ax4 = fig3.add_subplot(2, 3, 4)
        if 'cone_fit' in estimator.results and estimator.results['cone_fit'].get('outlier_coords') is not None:
            cone_fit = estimator.results['cone_fit']
            outlier_coords = cone_fit['outlier_coords']
            inlier_coords = cone_fit['inlier_coords']
            
            if len(outlier_coords['x']) > 0 and len(inlier_coords['x']) > 0:
                # Plot Z-distribution comparison
                ax4.hist(inlier_coords['z'], bins=30, alpha=0.6, label='Inliers', color='green', density=True)
                ax4.hist(outlier_coords['z'], bins=30, alpha=0.6, label='Outliers', color='red', density=True)
                ax4.set_xlabel('Z coordinate (mm)')
                ax4.set_ylabel('Density')
                ax4.set_title('Left: Z-Distribution (Inliers vs Outliers)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # 5. Residual analysis placeholder
        ax5 = fig3.add_subplot(2, 3, 5)
        if 'cone_fit' in estimator.results:
            cone_fit = estimator.results['cone_fit']
            ax5.text(0.5, 0.5, 'Left: Residual Analysis\n(To be implemented)', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Left: Residual Analysis')
        
        # 6. Summary statistics (like original estimator)
        ax6 = fig3.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = "LEFT SIDE ANALYSIS\n"
        summary_text += "="*40 + "\n"
        summary_text += f"Original points: {len(left_converted['x']):,}\n"
        
        if estimator.noise_filter:
            noise_removed = len(x_proc) - len(x_filtered)
            summary_text += f"Noise points removed: {noise_removed:,}\n"
        
        summary_text += f"Surface points: {len(surface_points['x']):,}\n"
        summary_text += f"Hole points: {len(hole_points['x']):,}\n\n"
        
        if 'cone_fit' in estimator.results:
            cone_fit = estimator.results['cone_fit']
            summary_text += "CONE FITTING:\n"
            summary_text += f"Half-angle: {cone_fit['half_angle_deg']:.1f}°\n"
            summary_text += f"RMSE: {cone_fit['rmse']:.4f} mm\n"
            summary_text += f"Outliers removed: {cone_fit.get('n_outliers_removed', 0)}\n"
            summary_text += f"Inliers used: {cone_fit.get('n_inliers_used', 0)}\n\n"
        
        if left_results['success']:
            summary_text += "DEPTH ESTIMATION:\n"
            summary_text += f"Total depth: {left_results['estimated_total_depth']:.3f} mm\n"
            summary_text += f"Apex depth: {left_results['estimated_apex_depth']:.3f} mm\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the figure
        plot_file = output_folder / "left_hole_detailed_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"     📊 Saved: {plot_file.name}")
        
        # Show the plot
        plt.show()
        
        # 3. Left fitted cone with inliers/outliers
        ax3 = fig3.add_subplot(2, 3, 3, projection='3d')
        if hasattr(estimator, 'results') and 'cone_fit' in estimator.results:
            cone_fit = estimator.results['cone_fit']
            
            # Plot inliers and outliers with clear distinction
            if cone_fit.get('outlier_coords') is not None and cone_fit.get('inlier_coords') is not None:
                inlier_coords = cone_fit['inlier_coords']
                outlier_coords = cone_fit['outlier_coords']
                
                if len(inlier_coords['x']) > 0:
                    ax3.scatter(inlier_coords['x'], inlier_coords['y'], inlier_coords['z'], 
                               c='green', s=2.5, alpha=0.8, label=f'Inliers ({len(inlier_coords["x"]):,})')
                
                if len(outlier_coords['x']) > 0:
                    ax3.scatter(outlier_coords['x'], outlier_coords['y'], outlier_coords['z'], 
                               c='red', s=4, alpha=0.9, label=f'Outliers ({len(outlier_coords["x"]):,})')
            
            # Plot fitted cone
            apex = cone_fit['apex']
            axis = cone_fit['axis']
            half_angle = cone_fit['half_angle_rad']
            
            self.plot_fitted_cone(ax3, apex, axis, half_angle, cone_height=2.0)
            
            ax3.legend(fontsize=9)
        else:
            ax3.scatter(left_converted['x'], left_converted['y'], left_converted['z'], 
                       c='gray', s=1, alpha=0.6)
        
        ax3.set_title(f'Left Side - Fitted Cone\nDepth: {left_results["estimated_total_depth"]:.3f} mm')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        
        # 4. Left segmentation statistics
        ax4 = fig3.add_subplot(2, 3, 4)
        if hasattr(estimator, 'surface_points') and hasattr(estimator, 'hole_points'):
            categories = ['Surface', 'Hole']
            counts = [len(estimator.surface_points['x']), len(estimator.hole_points['x'])]
            colors = ['lightblue', 'darkblue']
            
            bars = ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
            ax4.set_ylabel('Point Count')
            ax4.set_title('Left Side - Point Distribution')
            ax4.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            # Add RMSE info
            ax4.text(0.5, 0.85, f'RMSE: {left_results["fit_rmse"]:.3f} mm\nDepth: {left_results["estimated_total_depth"]:.3f} mm',
                    transform=ax4.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # 5. Left depth profile/histogram
        ax5 = fig3.add_subplot(2, 3, 5)
        if hasattr(estimator, 'hole_points') and len(estimator.hole_points['x']) > 0:
            ax5.hist(estimator.hole_points['z'], bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax5.axvline(left_results['estimated_total_depth'], color='red', linestyle='--', 
                       label=f'Depth: {left_results["estimated_total_depth"]:.3f} mm')
            ax5.set_xlabel('Z coordinate (mm)')
            ax5.set_ylabel('Point count')
            ax5.set_title('Left Side - Depth Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Left results summary
        ax6 = fig3.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""Left Side Results:

Depth: {left_results['estimated_total_depth']:.3f} mm
Apex Depth: {left_results['estimated_apex_depth']:.3f} mm
RMSE: {left_results['fit_rmse']:.3f} mm

Point Counts:
Total: {left_results['total_points']:,}
Surface: {left_results['surface_points']:,}
Hole: {left_results['hole_points']:,}

Cone Parameters:
Apex: ({left_results['cone_apex'][0]:.1f}, {left_results['cone_apex'][1]:.1f}, {left_results['cone_apex'][2]:.1f})
Half-angle: {np.degrees(left_results['cone_half_angle']):.1f}°
Optimization Cost: {left_results.get('optimization_cost', 'N/A')}"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax6.set_title('Left Side Summary', fontweight='bold')
        
        plt.tight_layout()
        
        # Save Figure 3
        plot_file3 = output_folder / "figure3_left_hole_analysis.png"
        plt.savefig(plot_file3, dpi=300, bbox_inches='tight')
        print(f"     📊 Saved: {plot_file3.name}")
        plt.show()
    def plot_plane_surface(self, ax, normal, point, size=10, alpha=0.3, color='cyan'):
        """Plot a plane surface for visualization"""
        # Create a mesh grid for the plane
        xx, yy = np.meshgrid(np.linspace(-size, size, 10), np.linspace(-size, size, 10))
        
        # Calculate z coordinates for the plane: normal • (r - point) = 0
        # z = (normal[0]*(point[0] - x) + normal[1]*(point[1] - y) + normal[2]*point[2]) / normal[2]
        if abs(normal[2]) > 1e-6:  # Avoid division by zero
            zz = (normal[0]*(point[0] - (point[0] + xx)) + 
                  normal[1]*(point[1] - (point[1] + yy)) + 
                  normal[2]*point[2]) / normal[2]
            
            # Translate to the actual point
            xx_world = point[0] + xx
            yy_world = point[1] + yy
            zz_world = point[2] + zz
            
            ax.plot_surface(xx_world, yy_world, zz_world, alpha=alpha, color=color)

    def plot_fitted_cone(self, ax, apex, axis, half_angle, cone_height=2.0, n_circles=10, n_points_per_circle=20):
        """Plot the fitted cone surface with proper geometry"""
        
        # Generate cone surface points
        heights = np.linspace(0.01, cone_height, n_circles)
        theta_circle = np.linspace(0, 2*np.pi, n_points_per_circle)
        
        for h in heights:
            radius = h * np.tan(half_angle)
            
            # Generate circle points
            circle_x = []
            circle_y = []
            circle_z = []
            
            for theta in theta_circle:
                # Create circle in local coordinate system
                local_x = radius * np.cos(theta)
                local_y = radius * np.sin(theta)
                local_z = 0
                
                # Transform to world coordinates
                z_axis = axis / np.linalg.norm(axis)
                
                # Find orthogonal vectors
                if abs(z_axis[2]) < 0.9:
                    temp_vec = np.array([0, 0, 1])
                else:
                    temp_vec = np.array([1, 0, 0])
                
                x_axis = np.cross(temp_vec, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                
                # Transform point
                world_point = (apex + h * z_axis + 
                              local_x * x_axis + local_y * y_axis)
                
                circle_x.append(world_point[0])
                circle_y.append(world_point[1])
                circle_z.append(world_point[2])
            
            # Close the circle
            circle_x.append(circle_x[0])
            circle_y.append(circle_y[0])
            circle_z.append(circle_z[0])
            
            # Plot circle
            ax.plot(circle_x, circle_y, circle_z, 'b-', alpha=0.6, linewidth=1)
        
        # Plot cone axis
        axis_end = apex + cone_height * (axis / np.linalg.norm(axis))
        ax.plot([apex[0], axis_end[0]], [apex[1], axis_end[1]], [apex[2], axis_end[2]], 
                'r-', linewidth=3, alpha=0.8, label='Cone Axis')
        
        # Plot apex
        ax.scatter([apex[0]], [apex[1]], [apex[2]], c='red', s=50, marker='*', 
                  label='Apex', alpha=1.0)

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Analyze a single PCD file for countersink depth")
    
    parser.add_argument('pcd_file', help='Path to the PCD file to analyze')
    parser.add_argument('-o', '--output', help='Output folder (default: auto-generated)')
    parser.add_argument('--save-intermediate', action='store_true', default=True,
                       help='Save intermediate processed PCD files')
    parser.add_argument('--no-save-intermediate', action='store_false', dest='save_intermediate',
                       help='Do not save intermediate files')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    # Estimator parameters
    parser.add_argument('--csk-angle', type=float, default=100.0,
                       help='Expected countersink angle (degrees)')
    parser.add_argument('--inner-radius', type=float, default=1.2446,
                       help='Expected inner radius (mm)')
    parser.add_argument('--optimizer', type=str, default='slsqp',
                       choices=['least_squares', 'slsqp'],
                       help='Optimization method')
    parser.add_argument('--noise-neighbors', type=int, default=100,
                       help='Minimum neighbors for noise filter')
    parser.add_argument('--noise-radius', type=float, default=0.5,
                       help='Search radius for noise filter (mm)')
    parser.add_argument('--outlier-method', type=str, default='percentile',
                       choices=['percentile', 'median_filter'],
                       help='Outlier removal method')
    parser.add_argument('--outlier-threshold', type=float, default=0.02,
                       help='Outlier threshold')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Prepare estimator parameters
    estimator_params = {
        'expected_csk_angle_deg': args.csk_angle,
        'expected_inner_radius': args.inner_radius,
        'optimizer': args.optimizer,
        'noise_neighbors': args.noise_neighbors,
        'noise_radius': args.noise_radius,
        'outlier_method': args.outlier_method,
        'outlier_threshold': args.outlier_threshold,
        'random_seed': args.random_seed
    }
    
    try:
        # Create analyzer
        analyzer = SinglePCDAnalyzer(
            estimator_params=estimator_params,
            save_intermediate=args.save_intermediate,
            generate_plots=args.plot
        )
        
        # Process the file
        results = analyzer.process_pcd_file(args.pcd_file, args.output)
        
        # Print summary
        left_results = results['left_results']
        right_results = results['right_results']
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        if left_results['success'] and right_results['success']:
            print(f"✅ Both sides analyzed successfully")
            print(f"   Left depth:  {left_results['estimated_total_depth']:.3f} mm")
            print(f"   Right depth: {right_results['estimated_total_depth']:.3f} mm")
            avg_depth = (left_results['estimated_total_depth'] + right_results['estimated_total_depth']) / 2
            print(f"   Average:     {avg_depth:.3f} mm")
        elif left_results['success']:
            print(f"✅ Left side analyzed successfully")
            print(f"   Left depth: {left_results['estimated_total_depth']:.3f} mm")
            print(f"❌ Right side failed")
        elif right_results['success']:
            print(f"✅ Right side analyzed successfully") 
            print(f"   Right depth: {right_results['estimated_total_depth']:.3f} mm")
            print(f"❌ Left side failed")
        else:
            print(f"❌ Both sides failed to analyze")
        
        print(f"\nResults saved to: {results['output_folder']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())