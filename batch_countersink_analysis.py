#!/usr/bin/env python3
"""
Batch Countersink Analysis Script
--------------------------------
Processes all PCD files in a dataset folder with the complete analysis pipeline:
1. Load PCD files
2. Convert from meters to mm and flip z-axis
3. Split into left/right halves
4. Run countersink depth estimation on each half
5. Generate comprehensive analysis reports

This script reuses existing functionality from:
- test_pcd_loading_and_splitting.py (for PCD loading and splitting)
- convert_pcd.py (for coordinate transformations)
- countersink_depth_estimator.py (for depth estimation)

Usage:
    python batch_countersink_analysis.py dataset/ [options]
    
Requirements:
    pip install numpy scipy matplotlib pandas
"""

import os
import sys
import glob
import argparse
import json
import csv
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

class BatchCountersinkAnalyzer:
    def __init__(self, 
                 dataset_folder: str,
                 output_folder: str = None,
                 estimator_params: Dict = None,
                 save_intermediate: bool = True,
                 generate_plots: bool = False,
                 analysis_method: str = 'separate'):
        """
        Initialize batch analyzer
        
        Args:
            dataset_folder: Path to folder containing PCD files
            output_folder: Path for output files (default: dataset_folder/analysis_results)
            estimator_params: Parameters for CountersinkDepthEstimator
            save_intermediate: Save intermediate processed PCD files
            generate_plots: Generate visualization plots for each analysis
            analysis_method: 'separate' (analyze left/right separately) or 'global' (global plane first)
        """
        self.dataset_folder = Path(dataset_folder)
        self.output_folder = Path(output_folder) if output_folder else self.dataset_folder / "analysis_results"
        self.save_intermediate = save_intermediate
        self.generate_plots = generate_plots
        self.analysis_method = analysis_method
        
        # Validate analysis method
        if self.analysis_method not in ['separate', 'global']:
            raise ValueError("analysis_method must be 'separate' or 'global'")
        
        # Create output directories
        self.output_folder.mkdir(exist_ok=True)
        if self.save_intermediate:
            (self.output_folder / "processed_pcds").mkdir(exist_ok=True)
        if self.generate_plots:
            (self.output_folder / "plots").mkdir(exist_ok=True)
        
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
            'noise_radius': 0.5     # Match countersink_depth_estimator.py default
        }
        
        if estimator_params:
            default_params.update(estimator_params)
        
        self.estimator_params = default_params
        
        # Results storage
        self.results = []
        
        print(f"🚀 Batch Countersink Analyzer initialized")
        print(f"   Dataset folder: {self.dataset_folder}")
        print(f"   Output folder: {self.output_folder}")
        print(f"   Analysis method: {self.analysis_method}")
        print(f"   Save intermediate files: {self.save_intermediate}")
        print(f"   Generate plots: {self.generate_plots}")
    
    def find_pcd_files(self) -> List[Path]:
        """Find all PCD files matching the pattern Hole-xxxx_3d.pcd"""
        pattern = "Hole-*_3d.pcd"
        pcd_files = list(self.dataset_folder.glob(pattern))
        
        if not pcd_files:
            print(f"⚠️  No PCD files found matching pattern '{pattern}' in {self.dataset_folder}")
            # Try broader pattern
            broad_pattern = "*.pcd"
            pcd_files = list(self.dataset_folder.glob(broad_pattern))
            if pcd_files:
                print(f"   Found {len(pcd_files)} PCD files with broader pattern '{broad_pattern}'")
            else:
                print(f"   No PCD files found at all in {self.dataset_folder}")
        
        pcd_files.sort()  # Sort for consistent processing order
        print(f"📁 Found {len(pcd_files)} PCD files to process")
        
        return pcd_files
    
    def extract_hole_number(self, filename: str) -> str:
        """Extract hole number from filename like 'Hole-1172_3d.pcd'"""
        try:
            # Extract number between 'Hole-' and '_3d'
            if 'Hole-' in filename and '_3d' in filename:
                start = filename.find('Hole-') + 5
                end = filename.find('_3d')
                return filename[start:end]
            else:
                # Fallback: use filename without extension
                return Path(filename).stem
        except:
            return Path(filename).stem
    
    def preprocess_pcd(self, pcd_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess PCD file: convert to mm and flip z-axis
        Reuses functionality from convert_pcd.py
        """
        print(f"  🔄 Preprocessing: {pcd_file.name}")
        
        # Load PCD using converter (handles robust loading)
        x, y, z, header_lines = self.pcd_converter.load_pcd_file(str(pcd_file))
        
        # Apply transformations: meters to mm + flip z
        x_proc, y_proc, z_proc = self.pcd_converter.apply_transformations(
            x, y, z, meters_to_mm=True, flip_z=True
        )
        
        print(f"     ✅ Preprocessed {len(x_proc):,} points")
        
        return x_proc, y_proc, z_proc
    
    def split_point_cloud(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[Dict, Dict, float]:
        """
        Split point cloud into left/right halves
        Reuses functionality from test_pcd_loading_and_splitting.py
        """
        print(f"  ✂️  Splitting point cloud...")
        
        # Use the PCDTester's split functionality
        left_points, right_points, split_line = self.pcd_tester.split_point_cloud(x, y, z)
        
        print(f"     Left: {len(left_points['x']):,} points")
        print(f"     Right: {len(right_points['x']):,} points")
        print(f"     Split line: x = {split_line:.6f}")
        
        return left_points, right_points, split_line
    
    def estimate_global_plane(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
        """
        Estimate surface plane parameters from entire point cloud
        Similar to single_pcd_analysis.py
        
        Args:
            x, y, z: Point cloud coordinates (already converted)
            
        Returns:
            Dictionary with global plane parameters
        """
        print(f"  🌐 Estimating global plane from {len(x):,} points...")
        
        # Create a temporary estimator to get plane parameters
        temp_estimator = CountersinkDepthEstimator(**self.estimator_params)
        
        # Apply noise filtering
        x_filtered, y_filtered, z_filtered, noise_mask = temp_estimator.filter_noise(x, y, z)
        
        print(f"     Points after noise filtering: {len(x_filtered):,}")
        
        # Estimate surface plane using the same method as the estimator
        surface_normal, surface_point, surface_threshold = temp_estimator.estimate_surface_plane(
            x_filtered, y_filtered, z_filtered
        )
        
        plane_params = {
            'normal': surface_normal,
            'point': surface_point,
            'threshold': surface_threshold,
            'filtered_points': len(x_filtered),
            'original_points': len(x)
        }
        
        print(f"     Global plane normal: ({surface_normal[0]:.6f}, {surface_normal[1]:.6f}, {surface_normal[2]:.6f})")
        print(f"     Global plane point: ({surface_point[0]:.3f}, {surface_point[1]:.3f}, {surface_point[2]:.3f})")
        print(f"     Surface thickness threshold: {surface_threshold:.6f} mm")
        
        return plane_params
    
    def save_split_pcds(self, left_points: Dict, right_points: Dict, hole_number: str):
        """Save split point clouds as separate PCD files"""
        if not self.save_intermediate:
            return
        
        output_dir = self.output_folder / "processed_pcds"
        
        # Save left half
        left_filename = output_dir / f"Hole-{hole_number}_left_processed.pcd"
        self.pcd_tester.save_pcd_half(left_points, str(left_filename), 'left')
        
        # Save right half
        right_filename = output_dir / f"Hole-{hole_number}_right_processed.pcd"
        self.pcd_tester.save_pcd_half(right_points, str(right_filename), 'right')
        
        print(f"     💾 Saved split PCDs: {left_filename.name}, {right_filename.name}")

        return left_filename, right_filename


    def estimate_depth(self, points: Dict, side: str) -> Dict:
        """
        Estimate countersink depth for one side
        Reuses functionality from countersink_depth_estimator.py
        """
        print(f"  🎯 Estimating depth for {side} side...")
        
        # Create estimator with configured parameters
        print("Estimator params: ", self.estimator_params)
        estimator = CountersinkDepthEstimator(**self.estimator_params)
        
        try:
            # Use the estimate_depth_from_arrays method directly
            results = estimator.estimate_depth_from_arrays(
                points['x'], points['y'], points['z']
            )            
            
            # Add side identification
            results['side'] = side
            results['success'] = True
            results['error_message'] = None
            
            print(f"     ✅ {side.capitalize()} depth: {results['estimated_total_depth']:.6f} mm")
            print(f"     RMSE: {results['fit_rmse']:.6f}")
            
            return results
            
        except Exception as e:
            print(f"     ❌ {side.capitalize()} estimation failed: {e}")
            
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
        

    def estimate_depth_from_file(self, filename: str, points: Dict, side: str) -> Dict:
        """
        Estimate countersink depth for one side
        Reuses functionality from countersink_depth_estimator.py
        """
        print(f"  🎯 Estimating depth for {side} side...")
        
        # Create estimator with configured parameters
        estimator = CountersinkDepthEstimator(**self.estimator_params)
        
        try:
            # Use the estimate_depth_from_arrays method directly
            results = estimator.estimate_depth(filename=str(filename))            
            
            
            # Add side identification
            results['side'] = side
            results['success'] = True
            results['error_message'] = None
            
            print(f"     ✅ {side.capitalize()} depth: {results['estimated_total_depth']:.6f} mm")
            print(f"     RMSE: {results['fit_rmse']:.6f}")
            
            return results
            
        except Exception as e:
            print(f"     ❌ {side.capitalize()} estimation failed: {e}")
            
            # Return failure result
            return {
                'side': side,
                'success': False,
                'error_message': str(e),
                'estimated_total_depth': np.nan,
                'estimated_apex_depth': np.nan,
                'fit_rmse': np.nan,
                # 'total_points': len(points['x']) if 'x' in points else 0,
                'surface_points': 0,
                'hole_points': 0
            }
    
    def estimate_depth_with_global_plane(self, points: Dict, side: str, global_plane_params: Dict) -> Dict:
        """
        Estimate countersink depth using global plane parameters
        Similar to single_pcd_analysis.py approach
        
        Args:
            points: Dictionary with 'x', 'y', 'z' arrays
            side: 'left' or 'right'
            global_plane_params: Global plane parameters to use instead of local estimation
            
        Returns:
            Dictionary with estimation results
        """
        print(f"  🎯 Estimating depth for {side} side with global plane...")
        
        # Create estimator with configured parameters
        estimator = CountersinkDepthEstimator(**self.estimator_params)
        
        try:
            # Use the estimate_depth_with_global_plane method (from single_pcd_analysis.py)
            results = estimator.estimate_depth_with_global_plane(
                points['x'], points['y'], points['z'], global_plane_params
            )
            
            # Add side identification
            results['side'] = side
            results['success'] = True
            results['error_message'] = None
            
            print(f"     ✅ {side.capitalize()} depth (global plane): {results['estimated_total_depth']:.6f} mm")
            print(f"     RMSE: {results['fit_rmse']:.6f}")
            
            return results
            
        except Exception as e:
            print(f"     ❌ {side.capitalize()} estimation with global plane failed: {e}")
            
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
            if depth_diff < 0.05:
                consistency_rating = 'excellent'
            elif depth_diff < 0.1:
                consistency_rating = 'good'
            elif depth_diff < 0.2:
                consistency_rating = 'fair'
            else:
                consistency_rating = 'poor'
            
            analysis.update({
                'depth_difference': depth_diff,
                'apex_difference': apex_diff,
                'relative_depth_difference_percent': relative_depth_diff,
                'relative_apex_difference_percent': relative_apex_diff,
                'average_total_depth': avg_depth,
                'average_apex_depth': avg_apex,
                'consistency_rating': consistency_rating,
                'recommended_depth': avg_depth
            })
            
        else:
            analysis.update({
                'depth_difference': np.nan,
                'apex_difference': np.nan,
                'relative_depth_difference_percent': np.nan,
                'relative_apex_difference_percent': np.nan,
                'average_total_depth': np.nan,
                'average_apex_depth': np.nan,
                'consistency_rating': 'failed',
                'recommended_depth': np.nan
            })
        
        return analysis
    
    def generate_visualization(self, hole_number: str, left_points: Dict, right_points: Dict, 
                             left_results: Dict, right_results: Dict, bilateral_analysis: Dict):
        """Generate visualization plots for the analysis"""
        if not self.generate_plots:
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Hole-{hole_number} Bilateral Analysis', fontsize=16)
            
            # 1. Combined point cloud
            ax1 = axes[0, 0]
            if len(left_points['x']) > 0:
                ax1.scatter(left_points['x'], left_points['y'], c='blue', s=1, alpha=0.6, label='Left')
            if len(right_points['x']) > 0:
                ax1.scatter(right_points['x'], right_points['y'], c='red', s=1, alpha=0.6, label='Right')
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Y (mm)')
            ax1.set_title('Split Point Cloud (Top View)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Depth comparison
            ax2 = axes[0, 1]
            if bilateral_analysis['both_successful']:
                sides = ['Left', 'Right']
                depths = [left_results['estimated_total_depth'], right_results['estimated_total_depth']]
                colors = ['blue', 'red']
                bars = ax2.bar(sides, depths, color=colors, alpha=0.7)
                ax2.set_ylabel('Depth (mm)')
                ax2.set_title('Depth Comparison')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, depth in zip(bars, depths):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{depth:.3f}', ha='center', va='bottom')
            else:
                ax2.text(0.5, 0.5, 'Analysis\nFailed', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Depth Comparison - Failed')
            
            # 3. Quality metrics
            ax3 = axes[0, 2]
            if bilateral_analysis['both_successful']:
                metrics = ['RMSE Left', 'RMSE Right', 'Depth Diff']
                values = [left_results['fit_rmse'], right_results['fit_rmse'], 
                         bilateral_analysis['depth_difference']]
                bars = ax3.bar(metrics, values, alpha=0.7)
                ax3.set_ylabel('Error (mm)')
                ax3.set_title('Quality Metrics')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Quality metrics\nnot available', ha='center', va='center', 
                        transform=ax3.transAxes)
                ax3.set_title('Quality Metrics - N/A')
            
            # 4. Z-distribution comparison
            ax4 = axes[1, 0]
            if len(left_points['z']) > 0 and len(right_points['z']) > 0:
                ax4.hist(left_points['z'], bins=30, alpha=0.6, label='Left', color='blue', density=True)
                ax4.hist(right_points['z'], bins=30, alpha=0.6, label='Right', color='red', density=True)
                ax4.set_xlabel('Z coordinate (mm)')
                ax4.set_ylabel('Density')
                ax4.set_title('Z-Distribution Comparison')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            # 5. Summary statistics
            ax5 = axes[1, 1]
            ax5.axis('off')
            
            summary_text = f"HOLE-{hole_number} SUMMARY\n"
            summary_text += "="*30 + "\n"
            summary_text += f"Left points: {len(left_points['x']):,}\n"
            summary_text += f"Right points: {len(right_points['x']):,}\n\n"
            
            if bilateral_analysis['both_successful']:
                summary_text += f"Left depth: {left_results['estimated_total_depth']:.3f} mm\n"
                summary_text += f"Right depth: {right_results['estimated_total_depth']:.3f} mm\n"
                summary_text += f"Average: {bilateral_analysis['average_total_depth']:.3f} mm\n"
                summary_text += f"Difference: {bilateral_analysis['depth_difference']:.3f} mm\n"
                summary_text += f"Consistency: {bilateral_analysis['consistency_rating'].upper()}\n"
            else:
                summary_text += f"Left: {'✅' if bilateral_analysis['left_successful'] else '❌'}\n"
                summary_text += f"Right: {'✅' if bilateral_analysis['right_successful'] else '❌'}\n"
                summary_text += "Analysis incomplete\n"
            
            ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # 6. Error analysis
            ax6 = axes[1, 2]
            if bilateral_analysis['both_successful']:
                error_types = ['Rel. Depth\nDiff (%)', 'Rel. Apex\nDiff (%)']
                error_values = [bilateral_analysis['relative_depth_difference_percent'],
                               bilateral_analysis['relative_apex_difference_percent']]
                bars = ax6.bar(error_types, error_values, alpha=0.7)
                ax6.set_ylabel('Relative Error (%)')
                ax6.set_title('Relative Error Analysis')
                ax6.grid(True, alpha=0.3)
                
                # Color bars based on error magnitude
                for bar, val in zip(bars, error_values):
                    if val < 5:
                        bar.set_color('green')
                    elif val < 10:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
            else:
                ax6.text(0.5, 0.5, 'Error analysis\nnot available', ha='center', va='center', 
                        transform=ax6.transAxes)
                ax6.set_title('Error Analysis - N/A')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = self.output_folder / "plots" / f"Hole-{hole_number}_analysis.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"     📊 Visualization saved: {plot_filename.name}")
            
        except Exception as e:
            print(f"     ⚠️  Failed to generate visualization: {e}")
    
    def process_single_pcd(self, pcd_file: Path) -> Dict:
        """Process a single PCD file through the complete pipeline"""
        hole_number = self.extract_hole_number(pcd_file.name)
        
        print(f"\n🔍 Processing: {pcd_file.name} (Hole-{hole_number}) - Method: {self.analysis_method}")
        self.pcd_converter = PCDConverter()
        self.pcd_tester = PCDTester(split_method='center', no_use_zone_percent=30)  # Use center split method
        
        try:
            if self.analysis_method == 'separate':
                return self._process_with_separate_analysis(pcd_file, hole_number)
            elif self.analysis_method == 'global':
                return self._process_with_global_analysis(pcd_file, hole_number)
            else:
                raise ValueError(f"Unknown analysis method: {self.analysis_method}")
                
        except Exception as e:
            print(f"❌ Failed processing {pcd_file.name}: {e}")
            
            # Return error result
            return {
                'timestamp': datetime.now().isoformat(),
                'file_info': {
                    'original_file': str(pcd_file),
                    'hole_number': hole_number,
                    'original_points': 0,
                    'left_points': 0,
                    'right_points': 0,
                    'split_line': np.nan
                },
                'processing': {
                    'preprocessing_applied': ['meters_to_mm', 'flip_z'],
                    'split_method': 'center',
                    'analysis_method': self.analysis_method,
                    'estimator_params': self.estimator_params
                },
                'left_results': {'side': 'left', 'success': False, 'error_message': str(e)},
                'right_results': {'side': 'right', 'success': False, 'error_message': str(e)},
                'bilateral_analysis': {'both_successful': False, 'error_message': str(e)}
            }
    
    def _process_with_separate_analysis(self, pcd_file: Path, hole_number: str) -> Dict:
        """Process PCD with separate left/right analysis (original method)"""
        print(f"  📋 Using separate analysis method")
        
        # Step 1: Load raw PCD (do NOT apply global conversion yet)
        print(f"  🔄 Loading raw PCD (no conversion yet): {pcd_file.name}")
        x_raw, y_raw, z_raw = self.pcd_tester.load_pcd_file(str(pcd_file))

        # Step 2: Split raw point cloud into left/right (split BEFORE conversion)
        left_raw, right_raw, split_line = self.pcd_tester.split_point_cloud(x_raw, y_raw, z_raw)
    
        print(f"  ✨ Converting left half ({len(left_raw['x'])} pts) to mm + flip Z")
        if len(left_raw['x']) > 0:
            lx, ly, lz = self.pcd_converter.apply_transformations(
                left_raw['x'], left_raw['y'], left_raw['z'], meters_to_mm=True, flip_z=True
            )
            left_points = {'x': lx, 'y': ly, 'z': lz}
        else:
            left_points = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}

        print(f"  ✨ Converting right half ({len(right_raw['x'])} pts) to mm + flip Z")
        if len(right_raw['x']) > 0:
            rx, ry, rz = self.pcd_converter.apply_transformations(
                right_raw['x'], right_raw['y'], right_raw['z'], meters_to_mm=True, flip_z=True
            )
            right_points = {'x': rx, 'y': ry, 'z': rz}
        else:
            right_points = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}

        # Step 3: Save intermediate processed PCDs (converted halves) if requested
        left_filename, right_filename = self.save_split_pcds(left_points, right_points, hole_number)

        # Step 4: Estimate depth for each side using converted halves (separate analysis)
        right_results = self.estimate_depth_from_file(right_filename, right_points, 'right')
        left_results = self.estimate_depth_from_file(left_filename, left_points, 'left')
        
        # Step 5: Analyze bilateral consistency
        bilateral_analysis = self.analyze_bilateral_consistency(left_results, right_results)
        
        # Step 6: Generate visualization
        self.generate_visualization(hole_number, left_points, right_points, 
                                  left_results, right_results, bilateral_analysis)
        
        # Compile comprehensive results
        result = {
            'timestamp': datetime.now().isoformat(),
            'file_info': {
                'original_file': str(pcd_file),
                'hole_number': hole_number,
                'original_points': len(x_raw),
                'left_points': len(left_points['x']),
                'right_points': len(right_points['x']),
                'split_line': split_line
            },
            'processing': {
                'preprocessing_applied': ['meters_to_mm', 'flip_z'],
                'split_method': 'center',
                'analysis_method': 'separate',
                'estimator_params': self.estimator_params
            },
            'left_results': left_results,
            'right_results': right_results,
            'bilateral_analysis': bilateral_analysis
        }
        
        print(f"✅ Completed: Hole-{hole_number} (separate analysis)")
        if bilateral_analysis['both_successful']:
            print(f"   Recommended depth: {bilateral_analysis['recommended_depth']:.6f} mm")
            print(f"   Consistency: {bilateral_analysis['consistency_rating']}")
        
        return result
    
    def _process_with_global_analysis(self, pcd_file: Path, hole_number: str) -> Dict:
        """Process PCD with global plane analysis first (new method)"""
        print(f"  🌐 Using global plane analysis method")
        
        # Step 1: Load raw PCD and convert to mm + flip Z first
        print(f"  🔄 Loading and converting entire PCD: {pcd_file.name}")
        x_raw, y_raw, z_raw = self.pcd_tester.load_pcd_file(str(pcd_file))
        
        # Convert entire point cloud first
        x_converted, y_converted, z_converted = self.pcd_converter.apply_transformations(
            x_raw, y_raw, z_raw, meters_to_mm=True, flip_z=True
        )
        print(f"     Converted {len(x_converted):,} points to mm and flipped Z")
        
        # Step 2: Estimate global plane from converted point cloud
        global_plane_params = self.estimate_global_plane(x_converted, y_converted, z_converted)
        
        # Step 3: Split converted point cloud into left/right
        print(f"  ✂️  Splitting converted point cloud...")
        left_points, right_points, split_line = self.pcd_tester.split_point_cloud(
            x_converted, y_converted, z_converted
        )
        
        print(f"     Left: {len(left_points['x']):,} points")
        print(f"     Right: {len(right_points['x']):,} points")
        print(f"     Split line: x = {split_line:.6f}")
        
        # Step 4: Save intermediate processed PCDs (converted halves) if requested
        left_filename, right_filename = self.save_split_pcds(left_points, right_points, hole_number)

        # Step 5: Estimate depth for each side using global plane parameters
        left_results = self.estimate_depth_with_global_plane(left_points, 'left', global_plane_params)
        right_results = self.estimate_depth_with_global_plane(right_points, 'right', global_plane_params)
        
        # Step 6: Analyze bilateral consistency
        bilateral_analysis = self.analyze_bilateral_consistency(left_results, right_results)
        
        # Step 7: Generate visualization
        self.generate_visualization(hole_number, left_points, right_points, 
                                  left_results, right_results, bilateral_analysis)
        
        # Compile comprehensive results
        result = {
            'timestamp': datetime.now().isoformat(),
            'file_info': {
                'original_file': str(pcd_file),
                'hole_number': hole_number,
                'original_points': len(x_raw),
                'left_points': len(left_points['x']),
                'right_points': len(right_points['x']),
                'split_line': split_line
            },
            'processing': {
                'preprocessing_applied': ['meters_to_mm', 'flip_z'],
                'split_method': 'center',
                'analysis_method': 'global',
                'global_plane_used': True,
                'estimator_params': self.estimator_params
            },
            'global_plane_params': global_plane_params,
            'left_results': left_results,
            'right_results': right_results,
            'bilateral_analysis': bilateral_analysis
        }
        
        print(f"✅ Completed: Hole-{hole_number} (global plane analysis)")
        if bilateral_analysis['both_successful']:
            print(f"   Recommended depth: {bilateral_analysis['recommended_depth']:.6f} mm")
            print(f"   Consistency: {bilateral_analysis['consistency_rating']}")
        
        return result
    
    def process_all_pcds(self) -> List[Dict]:
        """Process all PCD files in the dataset folder"""
        pcd_files = self.find_pcd_files()
        
        if not pcd_files:
            print("❌ No PCD files found to process")
            return []
        
        print(f"\n🚀 Starting batch processing of {len(pcd_files)} files...")
        
        results = []
        successful = 0
        failed = 0
        
        for i, pcd_file in enumerate(pcd_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(pcd_files)}: {pcd_file.name}")
            print(f"{'='*60}")
            
            result = self.process_single_pcd(pcd_file)
            results.append(result)
            
            # Track success/failure
            if result['bilateral_analysis'].get('both_successful', False):
                successful += 1
            else:
                failed += 1
            
            # Progress update
            progress = (i / len(pcd_files)) * 100
            print(f"\n📈 Progress: {i}/{len(pcd_files)} ({progress:.1f}%) - ✅ {successful} successful, ❌ {failed} failed")
        
        self.results = results
        
        print(f"\n🎉 Batch processing complete!")
        print(f"   Total files processed: {len(pcd_files)}")
        print(f"   Successful analyses: {successful}")
        print(f"   Failed analyses: {failed}")
        print(f"   Success rate: {(successful/len(pcd_files)*100):.1f}%")
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save comprehensive results to files"""
        print(f"\n💾 Saving results...")
        
        # Save simple JSON results in requested format
        simple_json = {}
        for result in results:
            hole_number = result['file_info']['hole_number']
            key = f"Hole_number_{hole_number}"
            
            # Extract depth values (use NaN if failed)
            left_depth = result['left_results'].get('estimated_total_depth', np.nan)
            right_depth = result['right_results'].get('estimated_total_depth', np.nan)
            
            # Convert NaN to null for JSON
            simple_json[key] = {
                "left_total_depth": left_depth if not np.isnan(left_depth) else None,
                "right_total_depth": right_depth if not np.isnan(right_depth) else None
            }
        
        # Save simple JSON
        simple_json_file = self.output_folder / "results.json"
        with open(simple_json_file, 'w') as f:
            json.dump(simple_json, f, indent=2)
        print(f"   📄 Results JSON: {simple_json_file}")
        
        # Save detailed JSON results
        json_file = self.output_folder / "detailed_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   📄 Detailed results: {json_file}")
        
        # Save summary CSV
        csv_file = self.output_folder / "summary_results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'hole_number', 'file_name', 'original_points', 'left_points', 'right_points',
                'left_success', 'left_depth_mm', 'left_rmse', 'right_success', 'right_depth_mm', 'right_rmse',
                'both_successful', 'depth_difference_mm', 'relative_difference_percent', 
                'average_depth_mm', 'consistency_rating', 'recommended_depth_mm'
            ])
            
            # Data rows
            for result in results:
                file_info = result['file_info']
                left = result['left_results']
                right = result['right_results']
                bilateral = result['bilateral_analysis']
                
                writer.writerow([
                    file_info['hole_number'],
                    Path(file_info['original_file']).name,
                    file_info['original_points'],
                    file_info['left_points'],
                    file_info['right_points'],
                    left.get('success', False),
                    left.get('estimated_total_depth', np.nan),
                    left.get('fit_rmse', np.nan),
                    right.get('success', False),
                    right.get('estimated_total_depth', np.nan),
                    right.get('fit_rmse', np.nan),
                    bilateral.get('both_successful', False),
                    bilateral.get('depth_difference', np.nan),
                    bilateral.get('relative_depth_difference_percent', np.nan),
                    bilateral.get('average_total_depth', np.nan),
                    bilateral.get('consistency_rating', 'failed'),
                    bilateral.get('recommended_depth', np.nan)
                ])
        
        print(f"   📊 Summary CSV: {csv_file}")
        
        # Save configuration
        config_file = self.output_folder / "analysis_config.json"
        config = {
            'timestamp': datetime.now().isoformat(),
            'dataset_folder': str(self.dataset_folder),
            'output_folder': str(self.output_folder),
            'estimator_params': self.estimator_params,
            'processing_settings': {
                'save_intermediate': self.save_intermediate,
                'generate_plots': self.generate_plots,
                'preprocessing': ['meters_to_mm', 'flip_z'],
                'split_method': 'center'
            },
            'files_processed': len(results)
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"   ⚙️  Configuration: {config_file}")
    
    def generate_summary_report(self, results: List[Dict]):
        """Generate a comprehensive summary report"""
        print(f"\n📋 Generating summary report...")
        
        # Calculate statistics
        successful_results = [r for r in results if r['bilateral_analysis'].get('both_successful', False)]
        
        if not successful_results:
            print("   ⚠️  No successful results to summarize")
            return
        
        # Extract depth values
        depths = [r['bilateral_analysis']['recommended_depth'] for r in successful_results]
        differences = [r['bilateral_analysis']['depth_difference'] for r in successful_results]
        relative_diffs = [r['bilateral_analysis']['relative_depth_difference_percent'] for r in successful_results]
        
        # Calculate summary statistics
        stats = {
            'total_files': len(results),
            'successful_analyses': len(successful_results),
            'success_rate_percent': (len(successful_results) / len(results)) * 100,
            'depth_statistics': {
                'mean_depth_mm': np.mean(depths),
                'std_depth_mm': np.std(depths),
                'min_depth_mm': np.min(depths),
                'max_depth_mm': np.max(depths),
                'median_depth_mm': np.median(depths)
            },
            'consistency_statistics': {
                'mean_difference_mm': np.mean(differences),
                'std_difference_mm': np.std(differences),
                'max_difference_mm': np.max(differences),
                'mean_relative_difference_percent': np.mean(relative_diffs),
                'max_relative_difference_percent': np.max(relative_diffs)
            }
        }
        
        # Generate report text
        report_file = self.output_folder / "summary_report.txt"
        with open(report_file, 'w') as f:
            f.write("BATCH COUNTERSINK ANALYSIS SUMMARY REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"  Dataset folder: {self.dataset_folder}\n")
            f.write(f"  Output folder: {self.output_folder}\n")
            f.write(f"  Total files processed: {stats['total_files']}\n")
            f.write(f"  Successful analyses: {stats['successful_analyses']}\n")
            f.write(f"  Success rate: {stats['success_rate_percent']:.1f}%\n\n")
            
            f.write("DEPTH ANALYSIS:\n")
            ds = stats['depth_statistics']
            f.write(f"  Mean depth: {ds['mean_depth_mm']:.4f} ± {ds['std_depth_mm']:.4f} mm\n")
            f.write(f"  Median depth: {ds['median_depth_mm']:.4f} mm\n")
            f.write(f"  Range: {ds['min_depth_mm']:.4f} to {ds['max_depth_mm']:.4f} mm\n")
            f.write(f"  Variation: {ds['std_depth_mm']:.4f} mm\n\n")
            
            f.write("BILATERAL CONSISTENCY:\n")
            cs = stats['consistency_statistics']
            f.write(f"  Mean difference: {cs['mean_difference_mm']:.4f} ± {cs['std_difference_mm']:.4f} mm\n")
            f.write(f"  Maximum difference: {cs['max_difference_mm']:.4f} mm\n")
            f.write(f"  Mean relative difference: {cs['mean_relative_difference_percent']:.2f}%\n")
            f.write(f"  Maximum relative difference: {cs['max_relative_difference_percent']:.2f}%\n\n")
            
            # Consistency rating distribution
            ratings = [r['bilateral_analysis']['consistency_rating'] for r in successful_results]
            rating_counts = {rating: ratings.count(rating) for rating in ['excellent', 'good', 'fair', 'poor']}
            
            f.write("CONSISTENCY RATING DISTRIBUTION:\n")
            for rating, count in rating_counts.items():
                percentage = (count / len(successful_results)) * 100
                f.write(f"  {rating.capitalize()}: {count} ({percentage:.1f}%)\n")
        
        print(f"   📊 Summary report: {report_file}")
        print(f"\n📈 Key Statistics:")
        print(f"   Success rate: {stats['success_rate_percent']:.1f}%")
        print(f"   Mean depth: {stats['depth_statistics']['mean_depth_mm']:.4f} ± {stats['depth_statistics']['std_depth_mm']:.4f} mm")
        print(f"   Mean bilateral difference: {stats['consistency_statistics']['mean_difference_mm']:.4f} mm")
    
    def save_simple_json_only(self, results: List[Dict], output_file: str = None):
        """Save only the simple JSON format requested by user"""
        simple_json = {}
        for result in results:
            hole_number = result['file_info']['hole_number']
            key = f"Hole_number_{hole_number}"
            
            # Extract depth values (use NaN if failed)
            left_depth = result['left_results'].get('estimated_total_depth', np.nan)
            right_depth = result['right_results'].get('estimated_total_depth', np.nan)
            
            # Convert NaN to null for JSON
            simple_json[key] = {
                "left_total_depth": left_depth if not np.isnan(left_depth) else None,
                "right_total_depth": right_depth if not np.isnan(right_depth) else None
            }
        
        # Determine output file
        if output_file is None:
            output_file = self.output_folder / "results.json"
        else:
            output_file = Path(output_file)
        
        # Save simple JSON
        with open(output_file, 'w') as f:
            json.dump(simple_json, f, indent=2)
        
        print(f"📄 Simple results saved to: {output_file}")
        return simple_json


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Batch Countersink Analysis for PCD Dataset')
    parser.add_argument('dataset_folder', help='Path to folder containing PCD files')
    parser.add_argument('--output-folder', help='Output folder for results (default: dataset_folder/analysis_results)')
    parser.add_argument('--csk-angle', type=float, default=100.0, 
                       help='Expected countersink angle (degrees)')
    parser.add_argument('--inner-radius', type=float, default=1.2446,
                       help='Expected inner radius (mm)')
    parser.add_argument('--optimizer', choices=['least_squares', 'slsqp'], default='slsqp',
                       help='Optimization method')
    parser.add_argument('--no-intermediate', action='store_true',
                       help='Do not save intermediate PCD files')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots for each analysis')
    parser.add_argument('--simple-json-only', action='store_true',
                       help='Output only the simple JSON format with hole numbers and depths')
    parser.add_argument('--output-json', help='Specify output JSON filename (only with --simple-json-only)')
    parser.add_argument('--noise-neighbors', type=int, default=100,
                       help='Minimum neighbors for noise filter')
    parser.add_argument('--noise-radius', type=float, default=0.5,
                       help='Search radius for noise filter (mm)')
    parser.add_argument('--analysis-method', choices=['separate', 'global'], default='separate',
                       help='Analysis method: separate (analyze left/right separately) or global (global plane first)')
    
    args = parser.parse_args()
    
    # Validate dataset folder
    if not os.path.exists(args.dataset_folder):
        print(f"❌ Dataset folder not found: {args.dataset_folder}")
        return 1
    
    # Configure estimator parameters
    estimator_params = {
        'expected_csk_angle_deg': args.csk_angle,
        'expected_inner_radius': args.inner_radius,
        'optimizer': args.optimizer,
        'noise_neighbors': args.noise_neighbors,
        'noise_radius': args.noise_radius
    }
    
    # Create analyzer
    analyzer = BatchCountersinkAnalyzer(
        dataset_folder=args.dataset_folder,
        output_folder=args.output_folder,
        estimator_params=estimator_params,
        save_intermediate=not args.no_intermediate,
        generate_plots=args.generate_plots,
        analysis_method=args.analysis_method
    )
    
    try:
        # Process all PCDs
        results = analyzer.process_all_pcds()
        
        if results:
            if args.simple_json_only:
                # Output only the simple JSON format
                output_file = args.output_json if args.output_json else 'hole_depths_simple.json'
                if not os.path.isabs(output_file):
                    output_file = os.path.join(analyzer.output_folder, output_file)
                analyzer.save_simple_json_only(results, output_file)
                print(f"🎉 Simple JSON results saved to: {output_file}")
            else:
                # Save full results
                analyzer.save_results(results)
                
                # Generate summary report
                analyzer.generate_summary_report(results)
                
                print(f"\n🎉 Batch analysis complete! Results saved to: {analyzer.output_folder}")
        else:
            print(f"❌ No results to save")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())