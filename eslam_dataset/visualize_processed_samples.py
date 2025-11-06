#!/usr/bin/env python3
"""
Visualize Processed Profile Dataset Samples
Loads the processed dataset and visualizes selected samples showing:
- The preprocessed profile data (X_train) 
- The corresponding target depths (y_train)
- Sample metadata

Author: GitHub Copilot
Date: November 1, 2025
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


class ProcessedDatasetVisualizer:
    """Class to visualize samples from processed profile dataset"""
    
    def __init__(self, processed_data_path: str):
        """
        Initialize visualizer
        
        Args:
            processed_data_path: Path to processed dataset pickle file
        """
        self.processed_data_path = Path(processed_data_path)
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.train_metadata = None
        self.val_metadata = None
        self.test_metadata = None
        self.scaler_X = None
        self.scaler_y = None
        self.uniform_samples = None
        self.min_angle = None
        self.max_angle = None
        
        self.load_data()
    
    def load_data(self):
        """Load processed dataset"""
        print(f"📂 Loading processed dataset from: {self.processed_data_path}")
        
        if not self.processed_data_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {self.processed_data_path}")
        
        with open(self.processed_data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Extract arrays
        self.X_train = self.data['X_train']
        self.y_train = self.data['y_train'] 
        self.X_val = self.data['X_val']
        self.y_val = self.data['y_val']
        self.X_test = self.data['X_test']
        self.y_test = self.data['y_test']
        
        # Extract metadata
        self.train_metadata = self.data['train_metadata']
        self.val_metadata = self.data['val_metadata']
        self.test_metadata = self.data['test_metadata']
        
        # Extract scalers
        self.scaler_X = self.data['scaler_X']
        self.scaler_y = self.data['scaler_y']
        
        # Extract processing parameters
        self.uniform_samples = self.data['uniform_samples']
        self.min_angle = self.data['min_angle']
        self.max_angle = self.data['max_angle']
        
        print(f"✅ Loaded processed dataset successfully!")
        print(f"📊 Dataset sizes:")
        print(f"  Training: {len(self.X_train)} samples")
        print(f"  Validation: {len(self.X_val) if self.X_val is not None else 0} samples")
        print(f"  Testing: {len(self.X_test) if self.X_test is not None else 0} samples")
        print(f"  Feature dimensions: {self.X_train.shape[1]} (uniform samples: {self.uniform_samples})")
        print(f"  Angle range: {self.min_angle}° to {self.max_angle}°")
    
    def get_dataset_info(self):
        """Print comprehensive dataset information"""
        print(f"\n{'='*80}")
        print(f"PROCESSED DATASET INFORMATION")
        print(f"{'='*80}")
        
        print(f"Processing parameters:")
        print(f"  Uniform samples per profile: {self.uniform_samples}")
        print(f"  Angle range: {self.min_angle}° to {self.max_angle}°")
        print(f"  Processing time: {self.data.get('processing_time', 'Unknown')}")
        
        print(f"\nDataset sizes:")
        print(f"  Training:   {len(self.X_train):6d} samples")
        print(f"  Validation: {len(self.X_val) if self.X_val is not None else 0:6d} samples") 
        print(f"  Testing:    {len(self.X_test) if self.X_test is not None else 0:6d} samples")
        
        print(f"\nData shapes:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  y_train: {self.y_train.shape}")
        
        # Check if data is normalized
        x_normalized = hasattr(self.scaler_X, 'mean_') and self.scaler_X.mean_ is not None
        y_normalized = hasattr(self.scaler_y, 'mean_') and self.scaler_y.mean_ is not None
        
        print(f"\nNormalization status:")
        print(f"  Features (X): {'✅ Normalized' if x_normalized else '❌ Not normalized'}")
        print(f"  Targets (y):  {'✅ Normalized' if y_normalized else '❌ Not normalized'}")
        
        if y_normalized:
            # Show original scale statistics
            original_y_train = self.scaler_y.inverse_transform(self.y_train)
            print(f"\nOriginal target statistics (mm):")
            print(f"  Left depths  - Mean: {np.mean(original_y_train[:, 0]):.4f}, Std: {np.std(original_y_train[:, 0]):.4f}")
            print(f"  Right depths - Mean: {np.mean(original_y_train[:, 1]):.4f}, Std: {np.std(original_y_train[:, 1]):.4f}")
            print(f"  Left range:  {np.min(original_y_train[:, 0]):.4f} to {np.max(original_y_train[:, 0]):.4f}")
            print(f"  Right range: {np.min(original_y_train[:, 1]):.4f} to {np.max(original_y_train[:, 1]):.4f}")
        else:
            print(f"\nTarget statistics:")
            print(f"  Left depths  - Mean: {np.mean(self.y_train[:, 0]):.4f}, Std: {np.std(self.y_train[:, 0]):.4f}")
            print(f"  Right depths - Mean: {np.mean(self.y_train[:, 1]):.4f}, Std: {np.std(self.y_train[:, 1]):.4f}")
            print(f"  Left range:  {np.min(self.y_train[:, 0]):.4f} to {np.max(self.y_train[:, 0]):.4f}")
            print(f"  Right range: {np.min(self.y_train[:, 1]):.4f} to {np.max(self.y_train[:, 1]):.4f}")
        
        # Show augmentation breakdown if metadata available
        if self.train_metadata:
            augment_counts = {}
            for meta in self.train_metadata:
                aug_type = meta.get('augmentation_type', 'original')
                augment_counts[aug_type] = augment_counts.get(aug_type, 0) + 1
            
            print(f"\nTraining set augmentation breakdown:")
            for aug_type, count in augment_counts.items():
                print(f"  {aug_type}: {count} samples ({count/len(self.train_metadata)*100:.1f}%)")
    
    def visualize_sample(self, dataset: str = 'train', sample_index: int = 0, 
                        show_original_scale: bool = True, save_path: str = None):
        """
        Visualize a single sample from the dataset
        
        Args:
            dataset: 'train', 'val', or 'test'
            sample_index: Index of sample to visualize
            show_original_scale: Whether to show targets in original scale
            save_path: Path to save the plot
        """
        # Get the appropriate dataset
        if dataset == 'train':
            X_data = self.X_train
            y_data = self.y_train
            metadata = self.train_metadata
            dataset_name = "Training"
        elif dataset == 'val':
            X_data = self.X_val
            y_data = self.y_val
            metadata = self.val_metadata
            dataset_name = "Validation"
        elif dataset == 'test':
            X_data = self.X_test
            y_data = self.y_test
            metadata = self.test_metadata
            dataset_name = "Testing"
        else:
            raise ValueError("Dataset must be 'train', 'val', or 'test'")
        
        if X_data is None or len(X_data) == 0:
            print(f"❌ {dataset_name} dataset is empty or not available")
            return
        
        if sample_index >= len(X_data):
            print(f"❌ Sample index {sample_index} is out of range. Dataset has {len(X_data)} samples.")
            return
        
        # Get sample data
        x_sample = X_data[sample_index]
        y_sample = y_data[sample_index]
        
        # Get metadata if available
        sample_metadata = None
        if metadata and sample_index < len(metadata):
            sample_metadata = metadata[sample_index]
        
        # Check if features (X) are normalized
        x_normalized = hasattr(self.scaler_X, 'mean_') and self.scaler_X.mean_ is not None
        
        # Convert features to original scale if possible
        x_original = None
        if x_normalized:
            x_original = self.scaler_X.inverse_transform(x_sample.reshape(1, -1))[0]
        
        # Convert targets to original scale if requested and possible
        y_display = y_sample.copy()
        y_normalized = hasattr(self.scaler_y, 'mean_') and self.scaler_y.mean_ is not None
        if show_original_scale and y_normalized:
            y_display = self.scaler_y.inverse_transform(y_sample.reshape(1, -1))[0]
            scale_label = "mm"
        else:
            scale_label = "normalized" if y_normalized else "original"
        
        # Create angle array for x-axis
        angles = np.linspace(self.min_angle, self.max_angle, len(x_sample))
        print("Raw y_sample (as stored):", y_sample)
        print("Display y_sample (after scaling):", y_display)
        
        # Determine number of plots based on available data
        if x_normalized and x_original is not None:
            # Show both normalized and original profile data
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Normalized profile data (X)
            ax1.plot(angles, x_sample, 'b-', linewidth=2, alpha=0.8, label='Normalized Profile')
            ax1.scatter(angles[::50], x_sample[::50], c='blue', s=20, alpha=0.6, zorder=5)
            ax1.set_xlabel('Angle (degrees)')
            ax1.set_ylabel('Depth (normalized units)')
            ax1.set_title(f'{dataset_name} Dataset - Sample {sample_index}: Normalized Profile Data (X)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Original scale profile data (X)
            ax2.plot(angles, x_original, 'g-', linewidth=2, alpha=0.8, label='Original Scale Profile')
            ax2.scatter(angles[::50], x_original[::50], c='green', s=20, alpha=0.6, zorder=5)
            ax2.set_xlabel('Angle (degrees)')
            ax2.set_ylabel('Depth (original units)')
            ax2.set_title(f'{dataset_name} Dataset - Sample {sample_index}: Original Scale Profile Data (X)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Target depths (y)
            target_names = ['Left CSK Depth', 'Right CSK Depth']
            colors = ['red', 'green']
            x_pos = [0, 1]
            
            bars = ax3.bar(x_pos, y_display, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            ax3.set_xlabel('Countersink Position')
            ax3.set_ylabel(f'Depth ({scale_label})')
            ax3.set_title(f'{dataset_name} Dataset - Sample {sample_index}: Target Depths (y)')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(target_names)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, y_display)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 4: Profile comparison (normalized vs original)
            ax4.plot(angles, x_sample, 'b-', linewidth=2, alpha=0.8, label='Normalized', zorder=2)
            
            # Scale original to fit on same plot for comparison
            x_original_scaled = (x_original - np.min(x_original)) / (np.max(x_original) - np.min(x_original))
            x_sample_scaled = (x_sample - np.min(x_sample)) / (np.max(x_sample) - np.min(x_sample))
            
            ax4.plot(angles, x_original_scaled, 'g--', linewidth=2, alpha=0.8, label='Original (scaled to [0,1])', zorder=1)
            ax4.set_xlabel('Angle (degrees)')
            ax4.set_ylabel('Normalized Depth (0-1 scale)')
            ax4.set_title(f'{dataset_name} Dataset - Sample {sample_index}: Profile Comparison')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Add sample info to first plot
            info_text = f"Sample Index: {sample_index}\n"
            info_text += f"Dataset: {dataset_name}\n"
            info_text += f"Features: {len(x_sample)} points\n"
            info_text += f"Angle Range: {self.min_angle}° to {self.max_angle}°\n"
            info_text += f"X Data: {'Normalized' if x_normalized else 'Original'}\n"
            info_text += f"Y Data: {'Normalized' if y_normalized else 'Original'}"
            
            if sample_metadata:
                info_text += f"\n\nHole: {sample_metadata.get('hole_name', 'N/A')}"
                info_text += f"\nAugmentation: {sample_metadata.get('augmentation_type', 'N/A')}"
                info_text += f"\nProfile ID: {sample_metadata.get('profile_id', 'N/A')}"
                info_text += f"\ndy: {sample_metadata.get('dy_mm', 'N/A')} mm"
            
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        else:
            # Show only available profile data (single plot)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Available profile data (X)
            profile_label = 'Normalized Profile' if x_normalized else 'Profile Data'
            ax1.plot(angles, x_sample, 'b-', linewidth=2, alpha=0.8, label=profile_label)
            ax1.scatter(angles[::50], x_sample[::50], c='blue', s=20, alpha=0.6, zorder=5)
            ax1.set_xlabel('Angle (degrees)')
            ax1.set_ylabel('Depth (preprocessed units)')
            ax1.set_title(f'{dataset_name} Dataset - Sample {sample_index}: Profile Data (X)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add sample info to plot
            info_text = f"Sample Index: {sample_index}\n"
            info_text += f"Dataset: {dataset_name}\n"
            info_text += f"Features: {len(x_sample)} points\n"
            info_text += f"Angle Range: {self.min_angle}° to {self.max_angle}°"
            
            if sample_metadata:
                info_text += f"\nHole: {sample_metadata.get('hole_name', 'N/A')}"
                info_text += f"\nAugmentation: {sample_metadata.get('augmentation_type', 'N/A')}"
                info_text += f"\nProfile ID: {sample_metadata.get('profile_id', 'N/A')}"
                info_text += f"\ndy: {sample_metadata.get('dy_mm', 'N/A')} mm"
            
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 2: Target depths (y) - same as before
            target_names = ['Left CSK Depth', 'Right CSK Depth']
            colors = ['red', 'green']
            x_pos = [0, 1]
            
            bars = ax2.bar(x_pos, y_display, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            ax2.set_xlabel('Countersink Position')
            ax2.set_ylabel(f'Depth ({scale_label})')
            ax2.set_title(f'{dataset_name} Dataset - Sample {sample_index}: Target Depths (y)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(target_names)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, y_display)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Add target info
        target_text = f"Left Depth: {y_display[0]:.6f} {scale_label}\n"
        target_text += f"Right Depth: {y_display[1]:.6f} {scale_label}\n"
        target_text += f"Depth Difference: {abs(y_display[1] - y_display[0]):.6f} {scale_label}"
        
        # Add to appropriate axis
        if x_normalized and x_original is not None:
            ax3.text(0.02, 0.98, target_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax2.text(0.02, 0.98, target_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")
        
        plt.show()
        
        # Print detailed information
        print(f"\n{'='*60}")
        print(f"SAMPLE {sample_index} DETAILS ({dataset_name.upper()} DATASET)")
        print(f"{'='*60}")
        print(f"Profile data shape: {x_sample.shape}")
        print(f"Target data shape: {y_sample.shape}")
        
        print(f"Profile statistics (stored/normalized):")
        print(f"  Min: {np.min(x_sample):.6f}")
        print(f"  Max: {np.max(x_sample):.6f}")
        print(f"  Mean: {np.mean(x_sample):.6f}")
        print(f"  Std: {np.std(x_sample):.6f}")
        
        if x_original is not None:
            print(f"Profile statistics (original scale):")
            print(f"  Min: {np.min(x_original):.6f}")
            print(f"  Max: {np.max(x_original):.6f}")
            print(f"  Mean: {np.mean(x_original):.6f}")
            print(f"  Std: {np.std(x_original):.6f}")
        
        print(f"Target depths ({scale_label}):")
        print(f"  Left:  {y_display[0]:.6f}")
        print(f"  Right: {y_display[1]:.6f}")
        print(f"  Difference: {abs(y_display[1] - y_display[0]):.6f}")
        
        if sample_metadata:
            print(f"Metadata:")
            for key, value in sample_metadata.items():
                if key not in ['original_x_range', 'original_z_range']:
                    print(f"  {key}: {value}")
    
    def compare_samples(self, dataset: str = 'train', sample_indices: List[int] = [0, 1, 2], 
                       show_original_scale: bool = True, save_path: str = None):
        """
        Compare multiple samples from the dataset
        
        Args:
            dataset: 'train', 'val', or 'test'
            sample_indices: List of sample indices to compare
            show_original_scale: Whether to show targets in original scale
            save_path: Path to save the plot
        """
        # Get the appropriate dataset
        if dataset == 'train':
            X_data = self.X_train
            y_data = self.y_train
            metadata = self.train_metadata
            dataset_name = "Training"
        elif dataset == 'val':
            X_data = self.X_val
            y_data = self.y_val
            metadata = self.val_metadata
            dataset_name = "Validation"
        elif dataset == 'test':
            X_data = self.X_test
            y_data = self.y_test
            metadata = self.test_metadata
            dataset_name = "Testing"
        else:
            raise ValueError("Dataset must be 'train', 'val', or 'test'")
        
        if X_data is None or len(X_data) == 0:
            print(f"❌ {dataset_name} dataset is empty or not available")
            return
        
        # Validate sample indices
        valid_indices = [idx for idx in sample_indices if idx < len(X_data)]
        if len(valid_indices) != len(sample_indices):
            print(f"⚠️ Some indices are out of range. Using valid indices: {valid_indices}")
        sample_indices = valid_indices
        
        if len(sample_indices) == 0:
            print(f"❌ No valid sample indices provided")
            return
        
        # Create angle array
        angles = np.linspace(self.min_angle, self.max_angle, self.uniform_samples)
        
        # Check if features (X) are normalized
        x_normalized = hasattr(self.scaler_X, 'mean_') and self.scaler_X.mean_ is not None
        y_normalized = hasattr(self.scaler_y, 'mean_') and self.scaler_y.mean_ is not None
        
        # Create subplots - adjust based on whether we can show original scale data
        n_samples = len(sample_indices)
        if x_normalized:
            # Show normalized, original, and targets (3 columns)
            fig, axes = plt.subplots(n_samples, 3, figsize=(18, 4*n_samples))
            col_titles = ['Normalized Profile', 'Original Scale Profile', 'Target Depths']
        else:
            # Show only available profile and targets (2 columns)
            fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
            col_titles = ['Profile Data', 'Target Depths']
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_samples))
        
        for i, (sample_idx, color) in enumerate(zip(sample_indices, colors)):
            x_sample = X_data[sample_idx]
            y_sample = y_data[sample_idx]
            
            # Convert to original scale if possible
            x_original = None
            if x_normalized:
                x_original = self.scaler_X.inverse_transform(x_sample.reshape(1, -1))[0]
            
            # Convert targets to original scale if requested
            y_display = y_sample.copy()
            if show_original_scale and y_normalized:
                y_display = self.scaler_y.inverse_transform(y_sample.reshape(1, -1))[0]
                scale_label = "mm"
            else:
                scale_label = "normalized" if y_normalized else "original"
            
            if x_normalized:
                # Plot normalized profile data
                axes[i, 0].plot(angles, x_sample, color=color, linewidth=2, alpha=0.8, 
                               label=f'Sample {sample_idx}')
                axes[i, 0].scatter(angles[::50], x_sample[::50], c=color, s=15, alpha=0.6)
                axes[i, 0].set_xlabel('Angle (degrees)')
                axes[i, 0].set_ylabel('Depth (normalized)')
                axes[i, 0].set_title(f'Sample {sample_idx}: Normalized Profile')
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 0].legend()
                
                # Plot original scale profile data
                axes[i, 1].plot(angles, x_original, color=color, linewidth=2, alpha=0.8, 
                               label=f'Sample {sample_idx}')
                axes[i, 1].scatter(angles[::50], x_original[::50], c=color, s=15, alpha=0.6)
                axes[i, 1].set_xlabel('Angle (degrees)')
                axes[i, 1].set_ylabel('Depth (original units)')
                axes[i, 1].set_title(f'Sample {sample_idx}: Original Scale Profile')
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].legend()
                
                # Plot target depths
                target_positions = [0, 1]
                bars = axes[i, 2].bar(target_positions, y_display, 
                                     color=['red', 'green'], alpha=0.7, edgecolor='black')
                axes[i, 2].set_xlabel('Countersink Position')
                axes[i, 2].set_ylabel(f'Depth ({scale_label})')
                axes[i, 2].set_title(f'Sample {sample_idx}: Target Depths')
                axes[i, 2].set_xticks(target_positions)
                axes[i, 2].set_xticklabels(['Left CSK', 'Right CSK'])
                axes[i, 2].grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, value in zip(bars, y_display):
                    height = bar.get_height()
                    axes[i, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
                
                # Add metadata to normalized plot
                if metadata and sample_idx < len(metadata):
                    meta = metadata[sample_idx]
                    info_text = f"Hole: {meta.get('hole_name', 'N/A')}\n"
                    info_text += f"Aug: {meta.get('augmentation_type', 'N/A')}"
                    
                    axes[i, 0].text(0.02, 0.98, info_text, transform=axes[i, 0].transAxes, 
                                  fontsize=8, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                
            else:
                # Plot profile data (only available version)
                axes[i, 0].plot(angles, x_sample, color=color, linewidth=2, alpha=0.8, 
                               label=f'Sample {sample_idx}')
                axes[i, 0].scatter(angles[::50], x_sample[::50], c=color, s=15, alpha=0.6)
                axes[i, 0].set_xlabel('Angle (degrees)')
                axes[i, 0].set_ylabel('Depth (preprocessed)')
                axes[i, 0].set_title(f'Sample {sample_idx}: Profile Data')
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 0].legend()
                
                # Plot target depths
                target_positions = [0, 1]
                bars = axes[i, 1].bar(target_positions, y_display, 
                                     color=['red', 'green'], alpha=0.7, edgecolor='black')
                axes[i, 1].set_xlabel('Countersink Position')
                axes[i, 1].set_ylabel(f'Depth ({scale_label})')
                axes[i, 1].set_title(f'Sample {sample_idx}: Target Depths')
                axes[i, 1].set_xticks(target_positions)
                axes[i, 1].set_xticklabels(['Left CSK', 'Right CSK'])
                axes[i, 1].grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, value in zip(bars, y_display):
                    height = bar.get_height()
                    axes[i, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
                
                # Add metadata to profile plot
                if metadata and sample_idx < len(metadata):
                    meta = metadata[sample_idx]
                    info_text = f"Hole: {meta.get('hole_name', 'N/A')}\n"
                    info_text += f"Aug: {meta.get('augmentation_type', 'N/A')}"
                    
                    axes[i, 0].text(0.02, 0.98, info_text, transform=axes[i, 0].transAxes, 
                                  fontsize=8, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Set column titles
        if x_normalized:
            for j, title in enumerate(col_titles):
                axes[0, j].text(0.5, 1.15, title, transform=axes[0, j].transAxes, 
                              fontsize=14, fontweight='bold', ha='center')
        else:
            for j, title in enumerate(col_titles):
                axes[0, j].text(0.5, 1.15, title, transform=axes[0, j].transAxes, 
                              fontsize=14, fontweight='bold', ha='center')
        
        plt.suptitle(f'{dataset_name} Dataset - Sample Comparison', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comparison plot saved to: {save_path}")
        
        plt.show()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Visualize processed profile dataset samples')
    parser.add_argument('--processed-data', 
                       default='processed_profile_datasets/processed_profile_datasets.pkl',
                       help='Path to processed dataset pickle file')
    parser.add_argument('--dataset', choices=['train', 'val', 'test'], default='train',
                       help='Dataset to visualize from')
    parser.add_argument('--sample-index', type=int, default=0,
                       help='Index of sample to visualize')
    parser.add_argument('--compare-samples', nargs='+', type=int, 
                       help='Multiple sample indices to compare')
    parser.add_argument('--info-only', action='store_true',
                       help='Only show dataset information without plotting')
    parser.add_argument('--original-scale', action='store_true', default=True,
                       help='Show targets in original scale (mm) if normalized')
    parser.add_argument('--save', help='Path to save the plot')
    
    args = parser.parse_args()
    
    print("PROCESSED PROFILE DATASET VISUALIZER")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = ProcessedDatasetVisualizer(args.processed_data)
    
    # Show dataset info
    visualizer.get_dataset_info()
    
    if args.info_only:
        return
    
    # Visualize samples
    if args.compare_samples:
        print(f"\n🔍 Comparing samples {args.compare_samples} from {args.dataset} dataset...")
        visualizer.compare_samples(
            dataset=args.dataset,
            sample_indices=args.compare_samples,
            show_original_scale=args.original_scale,
            save_path=args.save
        )
    else:
        print(f"\n🔍 Visualizing sample {args.sample_index} from {args.dataset} dataset...")
        visualizer.visualize_sample(
            dataset=args.dataset,
            sample_index=args.sample_index,
            show_original_scale=args.original_scale,
            save_path=args.save
        )


if __name__ == "__main__":
    main()