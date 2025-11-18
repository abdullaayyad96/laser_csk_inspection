#!/usr/bin/env python3
"""
Profile Neural Network Training Pipeline for Countersink Depth Estimation
Estimates left and right countersink depths from 2D profile scan data

Features:
- Loads augmented training, validation, and testing datasets
- Uniform angle sampling and interpolation preprocessing
- Configurable MLP architecture for dual output (left/right depths)
- Model saving and loading capabilities
- Preprocessed dataset storage for visualization

Author: GitHub Copilot
Date: November 1, 2025
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Please install: pip install torch scikit-learn")
    TORCH_AVAILABLE = False


class ProfileDatasetProcessor:
    """Class to handle profile dataset preparation for neural network training"""

    def __init__(self, augmented_train_dir: str = "augmented_dataset",
                 split_dataset_dir: str = "split_dataset",
                 uniform_samples: int = 2048,
                 min_angle: float = -8.0, 
                 max_angle: float = 8.0,
                 noise_min_neighbors: int = 5,
                 noise_radius: float = 0.05,
                 enable_noise_removal: bool = True):
        """
        Initialize profile dataset processor
        
        Args:
            augmented_train_dir: Directory containing augmented training dataset
            split_dataset_dir: Directory containing validation and test datasets
            uniform_samples: Number of uniform samples per profile
            min_angle: Minimum angle for uniform sampling (degrees)
            max_angle: Maximum angle for uniform sampling (degrees)
            noise_min_neighbors: Minimum neighbors for noise removal (including point itself)
            noise_radius: Radius in mm for noise removal vicinity search
            enable_noise_removal: Whether to apply noise removal preprocessing
        """
        self.augmented_train_dir = Path(augmented_train_dir)
        self.split_dataset_dir = Path(split_dataset_dir)
        self.uniform_samples = uniform_samples
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.noise_min_neighbors = noise_min_neighbors
        self.noise_radius = noise_radius
        self.enable_noise_removal = enable_noise_removal
        
        # Processed data will be stored here
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()  # Keep for backward compatibility
        
        # Custom target normalization parameters
        self.target_left_mean = None
        self.target_left_range = None
        self.target_right_mean = None
        self.target_right_range = None
        
        # Metadata
        self.train_metadata = None
        self.val_metadata = None
        self.test_metadata = None
        
        print(f"📁 Augmented train directory: {self.augmented_train_dir}")
        print(f"📁 Split dataset directory: {self.split_dataset_dir}")
        print(f"🎯 Uniform samples: {self.uniform_samples}")
        print(f"📐 Angle range: {self.min_angle}° to {self.max_angle}°")
        print(f"🧹 Noise removal: {'Enabled' if self.enable_noise_removal else 'Disabled'}")
        if self.enable_noise_removal:
            print(f"  Min neighbors: {self.noise_min_neighbors}, Radius: {self.noise_radius} mm")

    def load_datasets(self) -> None:
        """Load all required datasets (augmented train, validation, test)"""
        print("\n🔄 Loading datasets...")
        
        # Load augmented training dataset
        augmented_train_path = self.augmented_train_dir / "augmented_train_dataset.pkl"
        if augmented_train_path.exists():
            with open(augmented_train_path, 'rb') as f:
                self.train_profiles = pickle.load(f)
            print(f"✅ Loaded augmented training dataset: {len(self.train_profiles)} profiles")
        else:
            raise FileNotFoundError(f"Augmented training dataset not found: {augmented_train_path}")
        
        # Load validation dataset
        val_path = self.split_dataset_dir / "validation_dataset.pkl"
        if val_path.exists():
            with open(val_path, 'rb') as f:
                self.val_profiles = pickle.load(f)
            print(f"✅ Loaded validation dataset: {len(self.val_profiles)} profiles")
        else:
            raise FileNotFoundError(f"Validation dataset not found: {val_path}")
        
        # Load test dataset
        test_path = self.split_dataset_dir / "test_dataset.pkl"
        if test_path.exists():
            with open(test_path, 'rb') as f:
                self.test_profiles = pickle.load(f)
            print(f"✅ Loaded test dataset: {len(self.test_profiles)} profiles")
        else:
            raise FileNotFoundError(f"Test dataset not found: {test_path}")

    def remove_noise_points(self, x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
                           min_neighbors: int = 5, vicinity_radius: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove noise points that don't have enough neighbors within a specified radius
        
        Args:
            x_data: X coordinates
            y_data: Y coordinates  
            z_data: Z coordinates
            min_neighbors: Minimum number of neighbors required (including the point itself)
            vicinity_radius: Radius in mm to search for neighbors
            
        Returns:
            Tuple of filtered (x_data, y_data, z_data) arrays
        """
        if len(x_data) <= min_neighbors:
            # If we have very few points, keep them all
            return x_data, y_data, z_data
        
        # Create coordinate matrix for distance calculation
        coords = np.column_stack([x_data, y_data, z_data])
        n_points = len(coords)
        
        # Calculate pairwise distances between all points
        distances = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
        
        # Count neighbors within vicinity_radius for each point (including the point itself)
        neighbor_counts = np.sum(distances <= vicinity_radius, axis=1)
        
        # Keep points that have at least min_neighbors within vicinity
        keep_mask = neighbor_counts >= min_neighbors
        
        # Report filtering statistics
        removed_count = np.sum(~keep_mask)
        # if removed_count > 0:
        #     print(f"    Noise removal: {removed_count}/{n_points} points removed ({removed_count/n_points*100:.1f}%)")
        
        return x_data[keep_mask], y_data[keep_mask], z_data[keep_mask]

    def uniform_angle_sampling(self, x_data: np.ndarray, y_data: np.ndarray, 
                              z_data: np.ndarray) -> np.ndarray:
        """
        Uniformly sample profile data based on angles calculated from x and z coordinates
        
        In profile scanning geometry:
        - x_data: Lateral position across the scan line
        - y_data: Fixed position (profile slice position) 
        - z_data: Depth measurements
        - angle = arctan(x_lateral / abs(z_depth)) - angle from vertical axis
        
        Args:
            x_data: X coordinates (lateral position)
            y_data: Y coordinates (profile position)
            z_data: Z coordinates (depth measurements)
            
        Returns:
            Uniformly sampled depth profile
        """
        # Remove NaN values first
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data) | np.isnan(z_data))
        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        z_valid = z_data[valid_mask]
        
        if len(x_valid) < 3:
            return np.full(self.uniform_samples, np.nan)
        
        # Apply noise removal before angle calculation (if enabled)
        if self.enable_noise_removal:
            x_filtered, y_filtered, z_filtered = self.remove_noise_points(
                x_valid, y_valid, z_valid, 
                min_neighbors=self.noise_min_neighbors, 
                vicinity_radius=self.noise_radius
            )
        else:
            x_filtered, y_filtered, z_filtered = x_valid, y_valid, z_valid
        
        if len(x_filtered) < 3:
            if self.enable_noise_removal:
                print(f"    ⚠️  Too few points after noise removal: {len(x_filtered)}")
            return np.full(self.uniform_samples, np.nan)
        
        # Calculate angles from vertical axis
        # angle = arctan(x_lateral / abs(z_depth))
        z_abs = np.abs(z_filtered)
        # Avoid division by zero
        z_abs = np.where(z_abs < 1e-6, 1e-6, z_abs)
        angles = np.degrees(np.arctan(x_filtered / z_abs))
        
        # Create uniform angle grid
        uniform_angles = np.linspace(self.min_angle, self.max_angle, self.uniform_samples)
        
        # Sort data by angle for interpolation
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]
        z_sorted = z_filtered[sort_idx]
        
        # Interpolate z coordinates at uniform angles
        uniform_z = np.interp(uniform_angles, angles_sorted, z_sorted, 
                             left=np.nan, right=np.nan)
        
        return uniform_z
    
    def handle_nan_values(self, profile_data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        """
        Handle NaN values in profile data
        
        Args:
            profile_data: Array with potential NaN values
            method: 'interpolate', 'ignore_value', or 'remove'
            
        Returns:
            Processed profile data
        """
        if method == 'interpolate':
            if np.all(np.isnan(profile_data)):
                return np.zeros_like(profile_data)
            
            # Linear interpolation for NaN values
            valid_mask = ~np.isnan(profile_data)
            if np.sum(valid_mask) >= 2:
                indices = np.arange(len(profile_data))
                profile_data[~valid_mask] = np.interp(
                    indices[~valid_mask], 
                    indices[valid_mask], 
                    profile_data[valid_mask]
                )
            else:
                # If too few valid points, fill with median
                median_val = np.nanmedian(profile_data)
                profile_data[~valid_mask] = median_val if not np.isnan(median_val) else 0.0
                
        elif method == 'ignore_value':
            # Replace NaN with a specific ignore value
            profile_data[np.isnan(profile_data)] = -999.0
            
        elif method == 'remove':
            # Remove NaN values (changes array size)
            profile_data = profile_data[~np.isnan(profile_data)]
        
        return profile_data

    def process_profile(self, profile: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Process a single profile for neural network input
        
        Args:
            profile: Profile dictionary with x_array_mm, y_array_mm, z_array_mm
            
        Returns:
            Tuple of (processed_input, target_depths, metadata)
        """
        # Extract coordinates
        x_data = profile['x_array_mm']
        y_data = profile['y_array_mm']
        z_data = profile['z_array_mm']
        
        # Apply uniform angle sampling
        uniform_z = self.uniform_angle_sampling(x_data, y_data, z_data)
        
        # Handle NaN values
        processed_z = self.handle_nan_values(uniform_z, method='interpolate')
        
        # Extract target depths (left and right countersink depths)
        left_depth = profile['left_csk_depth_mm']
        right_depth = profile['right_csk_depth_mm']
        target_depths = np.array([left_depth, right_depth])
        
        # Create metadata
        metadata = {
            'hole_name': profile['hole_name'],
            'hole_number': profile['hole_number'],
            'profile_id': profile['profile_id'],
            'dy_mm': profile['dy_mm'],
            'num_points': profile['num_points'],
            'augmentation_type': profile.get('augmentation_type', 'original'),
            'augmentation_params': profile.get('augmentation_params', {}),
            'original_x_range': [np.nanmin(x_data), np.nanmax(x_data)],
            'original_z_range': [np.nanmin(z_data), np.nanmax(z_data)]
        }
        
        return processed_z, target_depths, metadata

    def prepare_datasets(self, nan_method: str = 'interpolate', 
                        normalize_features: bool = True, 
                        normalize_targets: bool = True) -> None:
        """
        Prepare all datasets for neural network training
        
        Args:
            nan_method: Method for handling NaN values
            normalize_features: Whether to normalize input features
            normalize_targets: Whether to normalize target values
        """
        print(f"\n🚀 Preparing datasets for neural network training...")
        print(f"🔧 NaN handling method: {nan_method}")
        print(f"📊 Feature normalization: {normalize_features}")
        print(f"🎯 Target normalization: {normalize_targets}")
        
        # Process training dataset
        print(f"\n🔄 Processing training dataset...")
        train_features = []
        train_targets = []
        train_metadata = []
        failed_train = 0
        
        for i, profile in enumerate(self.train_profiles):
            try:
                processed_input, target_depths, metadata = self.process_profile(profile)
                
                # Check for valid targets
                if np.any(np.isnan(target_depths)):
                    failed_train += 1
                    continue
                
                train_features.append(processed_input)
                train_targets.append(target_depths)
                train_metadata.append(metadata)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(self.train_profiles)} training profiles...")
                    
            except Exception as e:
                failed_train += 1
                print(f"  ⚠️  Failed to process training profile {i}: {e}")
        
        # Process validation dataset
        print(f"\n🔄 Processing validation dataset...")
        val_features = []
        val_targets = []
        val_metadata = []
        failed_val = 0
        
        for i, profile in enumerate(self.val_profiles):
            try:
                processed_input, target_depths, metadata = self.process_profile(profile)
                
                if np.any(np.isnan(target_depths)):
                    failed_val += 1
                    continue
                
                val_features.append(processed_input)
                val_targets.append(target_depths)
                val_metadata.append(metadata)
                
            except Exception as e:
                failed_val += 1
                print(f"  ⚠️  Failed to process validation profile {i}: {e}")
        
        # Process test dataset
        print(f"\n🔄 Processing test dataset...")
        test_features = []
        test_targets = []
        test_metadata = []
        failed_test = 0
        
        for i, profile in enumerate(self.test_profiles):
            try:
                processed_input, target_depths, metadata = self.process_profile(profile)
                
                if np.any(np.isnan(target_depths)):
                    failed_test += 1
                    continue
                
                test_features.append(processed_input)
                test_targets.append(target_depths)
                test_metadata.append(metadata)
                
            except Exception as e:
                failed_test += 1
                print(f"  ⚠️  Failed to process test profile {i}: {e}")
        
        # Convert to numpy arrays
        if len(train_features) == 0:
            raise ValueError("No valid training samples after processing")
        
        self.X_train = np.array(train_features)
        self.y_train = np.array(train_targets)
        self.train_metadata = train_metadata
        
        if len(val_features) > 0:
            self.X_val = np.array(val_features)
            self.y_val = np.array(val_targets)
            self.val_metadata = val_metadata
        
        if len(test_features) > 0:
            self.X_test = np.array(test_features)
            self.y_test = np.array(test_targets)
            self.test_metadata = test_metadata
        
        print(f"\n📊 Dataset preparation complete!")
        print(f"  Training: {len(self.X_train)} samples, {failed_train} failed")
        print(f"  Validation: {len(self.X_val) if self.X_val is not None else 0} samples, {failed_val} failed")
        print(f"  Testing: {len(self.X_test) if self.X_test is not None else 0} samples, {failed_test} failed")
        print(f"  Feature shape: {self.X_train.shape}")
        print(f"  Target shape: {self.y_train.shape}")
        
        # Display target statistics
        print(f"\n🎯 Target statistics:")
        print(f"  Left depths - Mean: {np.mean(self.y_train[:, 0]):.4f}, Std: {np.std(self.y_train[:, 0]):.4f}")
        print(f"  Right depths - Mean: {np.mean(self.y_train[:, 1]):.4f}, Std: {np.std(self.y_train[:, 1]):.4f}")
        print(f"  Left range: {np.min(self.y_train[:, 0]):.4f} to {np.max(self.y_train[:, 0]):.4f}")
        print(f"  Right range: {np.min(self.y_train[:, 1]):.4f} to {np.max(self.y_train[:, 1]):.4f}")
        
        # Normalization
        if normalize_features:
            print(f"\n🔄 Normalizing features across both samples and features...")
            # Calculate global mean and std across entire training feature matrix
            global_mean = np.mean(self.X_train)
            global_std = np.std(self.X_train)
            print(f"   Global mean: {global_mean:.6f}, Global std: {global_std:.6f}")
            
            # Apply global normalization to all datasets
            self.X_train = (self.X_train - global_mean) / global_std
            if self.X_val is not None:
                self.X_val = (self.X_val - global_mean) / global_std
            if self.X_test is not None:
                self.X_test = (self.X_test - global_mean) / global_std
            
            # Store normalization parameters for later use
            self.feature_global_mean = global_mean
            self.feature_global_std = global_std
            print(f"   ✅ Applied global feature normalization")
        
        if normalize_targets:
            print(f"🔄 Normalizing targets using (value - mean) / range...")
            # Calculate mean and range for each target (left and right depths)
            left_mean = np.mean(self.y_train[:, 0])
            left_min = np.min(self.y_train[:, 0])
            left_max = np.max(self.y_train[:, 0])
            left_range = left_max - left_min
            print(f"   Left depth  - Mean: {left_mean:.6f}, Range: {left_range:.6f} (min: {left_min:.6f}, max: {left_max:.6f})")
            left_std = 0.1 * np.std(self.y_train[:, 0])
            left_range = left_std
            
            right_mean = np.mean(self.y_train[:, 1])
            right_min = np.min(self.y_train[:, 1])
            right_max = np.max(self.y_train[:, 1])
            right_range = right_max - right_min
            print(f"   Right depth - Mean: {right_mean:.6f}, Range: {right_range:.6f} (min: {right_min:.6f}, max: {right_max:.6f})")
            right_std = 0.1 * np.std(self.y_train[:, 1])
            right_range = right_std
            
            print(f"   Left depth  - Mean: {left_mean:.6f}, Range: {left_range:.6f} (min: {left_min:.6f}, max: {left_max:.6f})")
            print(f"   Right depth - Mean: {right_mean:.6f}, Range: {right_range:.6f} (min: {right_min:.6f}, max: {right_max:.6f})")
            
            # Apply custom normalization: (value - mean) / range
            self.y_train[:, 0] = (self.y_train[:, 0] - left_mean) / left_range
            self.y_train[:, 1] = (self.y_train[:, 1] - right_mean) / right_range
            
            if self.y_val is not None:
                self.y_val[:, 0] = (self.y_val[:, 0] - left_mean) / left_range
                self.y_val[:, 1] = (self.y_val[:, 1] - right_mean) / right_range
                
            if self.y_test is not None:
                self.y_test[:, 0] = (self.y_test[:, 0] - left_mean) / left_range
                self.y_test[:, 1] = (self.y_test[:, 1] - right_mean) / right_range
            
            # Store normalization parameters for later use
            self.target_left_mean = left_mean
            self.target_left_range = left_range
            self.target_right_mean = right_mean
            self.target_right_range = right_range
            print(f"   ✅ Applied custom target normalization")

    def save_processed_datasets(self, output_dir: str = "processed_profile_datasets") -> None:
        """Save processed datasets for later use and visualization"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving processed datasets to: {output_path}")
        
        # Create comprehensive dataset dictionary
        processed_data = {
            # Arrays
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'y_val': self.y_val,
            'X_test': self.X_test,
            'y_test': self.y_test,
            
            # Metadata
            'train_metadata': self.train_metadata,
            'val_metadata': self.val_metadata,
            'test_metadata': self.test_metadata,
            
            # Scalers and normalization parameters
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_global_mean': getattr(self, 'feature_global_mean', None),
            'feature_global_std': getattr(self, 'feature_global_std', None),
            'target_left_mean': getattr(self, 'target_left_mean', None),
            'target_left_range': getattr(self, 'target_left_range', None),
            'target_right_mean': getattr(self, 'target_right_mean', None),
            'target_right_range': getattr(self, 'target_right_range', None),
            
            # Processing parameters
            'uniform_samples': self.uniform_samples,
            'min_angle': self.min_angle,
            'max_angle': self.max_angle,
            'noise_min_neighbors': self.noise_min_neighbors,
            'noise_radius': self.noise_radius,
            'enable_noise_removal': self.enable_noise_removal,
            
            # Dataset info
            'processing_time': datetime.now().isoformat(),
            'dataset_sizes': {
                'train': len(self.X_train),
                'val': len(self.X_val) if self.X_val is not None else 0,
                'test': len(self.X_test) if self.X_test is not None else 0
            }
        }
        
        # Save as pickle
        pickle_file = output_path / "processed_profile_datasets.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"  🥒 Saved pickle: {pickle_file.name}")
        
        # Save summary as text
        summary_file = output_path / "dataset_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PROCESSED PROFILE DATASETS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Processing timestamp: {processed_data['processing_time']}\n")
            f.write(f"Uniform samples per profile: {self.uniform_samples}\n")
            f.write(f"Angle range: {self.min_angle}° to {self.max_angle}°\n")
            f.write(f"Noise removal: {'Enabled' if self.enable_noise_removal else 'Disabled'}\n")
            if self.enable_noise_removal:
                f.write(f"  Min neighbors: {self.noise_min_neighbors}, Radius: {self.noise_radius} mm\n")
            f.write("\n")
            
            f.write("DATASET SIZES\n")
            f.write("-" * 15 + "\n")
            f.write(f"Training:   {len(self.X_train):6d} samples\n")
            f.write(f"Validation: {len(self.X_val) if self.X_val is not None else 0:6d} samples\n")
            f.write(f"Testing:    {len(self.X_test) if self.X_test is not None else 0:6d} samples\n\n")
            
            f.write("FEATURE INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Input features: {self.X_train.shape[1]}\n")
            f.write(f"Output targets: {self.y_train.shape[1]} (left and right depths)\n\n")
            
            f.write("TARGET STATISTICS (Training Set)\n")
            f.write("-" * 35 + "\n")
            f.write(f"Left depth  - Mean: {np.mean(self.y_train[:, 0]):.6f}, Std: {np.std(self.y_train[:, 0]):.6f}\n")
            f.write(f"Right depth - Mean: {np.mean(self.y_train[:, 1]):.6f}, Std: {np.std(self.y_train[:, 1]):.6f}\n")
            f.write(f"Left range:  {np.min(self.y_train[:, 0]):.6f} to {np.max(self.y_train[:, 0]):.6f}\n")
            f.write(f"Right range: {np.min(self.y_train[:, 1]):.6f} to {np.max(self.y_train[:, 1]):.6f}\n")
        
        print(f"  📋 Saved summary: {summary_file.name}")

    @classmethod
    def load_processed_datasets(cls, processed_data_path: str) -> 'ProfileDatasetProcessor':
        """Load previously processed datasets"""
        processor = cls.__new__(cls)
        
        with open(processed_data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Restore arrays
        processor.X_train = data['X_train']
        processor.y_train = data['y_train']
        processor.X_val = data['X_val']
        processor.y_val = data['y_val']
        processor.X_test = data['X_test']
        processor.y_test = data['y_test']
        
        # Restore metadata
        processor.train_metadata = data['train_metadata']
        processor.val_metadata = data['val_metadata']
        processor.test_metadata = data['test_metadata']
        
        # Restore scalers and normalization parameters
        processor.scaler_X = data['scaler_X']
        processor.scaler_y = data['scaler_y']
        processor.feature_global_mean = data.get('feature_global_mean', None)
        processor.feature_global_std = data.get('feature_global_std', None)
        processor.target_left_mean = data.get('target_left_mean', None)
        processor.target_left_range = data.get('target_left_range', None)
        processor.target_right_mean = data.get('target_right_mean', None)
        processor.target_right_range = data.get('target_right_range', None)
        
        # Restore parameters
        processor.uniform_samples = data['uniform_samples']
        processor.min_angle = data['min_angle']
        processor.max_angle = data['max_angle']
        
        # Restore noise removal parameters (with backward compatibility)
        processor.noise_min_neighbors = data.get('noise_min_neighbors', 5)
        processor.noise_radius = data.get('noise_radius', 0.05)
        processor.enable_noise_removal = data.get('enable_noise_removal', True)
        
        print(f"✅ Loaded processed datasets from: {processed_data_path}")
        print(f"📊 Training: {len(processor.X_train)}, Validation: {len(processor.X_val) if processor.X_val is not None else 0}, Testing: {len(processor.X_test) if processor.X_test is not None else 0}")
        
        return processor
    
    def normalize_features_global(self, X: np.ndarray) -> np.ndarray:
        """
        Apply global feature normalization to new data using stored parameters
        
        Args:
            X: Feature array to normalize
            
        Returns:
            Normalized feature array
        """
        if not hasattr(self, 'feature_global_mean') or not hasattr(self, 'feature_global_std'):
            raise ValueError("Global normalization parameters not available. Run prepare_datasets first with normalize_features=True")
        
        if self.feature_global_mean is None or self.feature_global_std is None:
            raise ValueError("Global normalization parameters are None. Features were not normalized during training")
        
        return (X - self.feature_global_mean) / self.feature_global_std
    
    def denormalize_features_global(self, X_normalized: np.ndarray) -> np.ndarray:
        """
        Reverse global feature normalization
        
        Args:
            X_normalized: Normalized feature array
            
        Returns:
            Original scale feature array
        """
        if not hasattr(self, 'feature_global_mean') or not hasattr(self, 'feature_global_std'):
            raise ValueError("Global normalization parameters not available")
        
        if self.feature_global_mean is None or self.feature_global_std is None:
            raise ValueError("Global normalization parameters are None")
        
        return X_normalized * self.feature_global_std + self.feature_global_mean

    def denormalize_targets(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Reverse custom target normalization: original = normalized * range + mean
        
        Args:
            y_normalized: Normalized target array (N, 2) for [left_depth, right_depth]
            
        Returns:
            Original scale target array
        """
        if not hasattr(self, 'target_left_mean') or not hasattr(self, 'target_left_range'):
            raise ValueError("Target normalization parameters not available")
        
        if (self.target_left_mean is None or self.target_left_range is None or 
            self.target_right_mean is None or self.target_right_range is None):
            raise ValueError("Target normalization parameters are None")
        
        y_denormalized = y_normalized.copy()
        
        # Reverse normalization: original = (normalized * range) + mean
        y_denormalized[:, 0] = (y_normalized[:, 0] * self.target_left_range) + self.target_left_mean
        y_denormalized[:, 1] = (y_normalized[:, 1] * self.target_right_range) + self.target_right_mean
        
        return y_denormalized


class DualOutputMLP(nn.Module):
    """Configurable Multi-Layer Perceptron for dual output regression (left and right depths)"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], 
                 activation: str = 'relu', dropout_rate: float = 0.2, 
                 batch_norm: bool = True, use_conv: bool = False, 
                 conv_channels: int = 1, conv_kernel_size: int = 50,
                 output_mode: str = 'both', use_classification: bool = False,
                 bin_width: float = 0.005, depth_range: tuple = (0.0, 5.0)):
        """
        Initialize dual output MLP with optional convolutional layer
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            dropout_rate: Dropout rate (0 to disable)
            batch_norm: Whether to use batch normalization
            use_conv: Whether to use a convolutional layer as the first layer
            conv_channels: Number of output channels for the convolutional layer
            conv_kernel_size: Kernel size for the convolutional layer (width)
            output_mode: Which outputs to train ('left', 'right', 'both')
            use_classification: Whether to use classification with binning instead of regression
            bin_width: Width of each depth bin in mm (for classification mode)
            depth_range: (min_depth, max_depth) range for binning
        """
        super(DualOutputMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = batch_norm
        self.use_conv = use_conv
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.output_mode = output_mode
        self.use_classification = use_classification
        self.bin_width = bin_width
        self.depth_range = depth_range
        
        # Validate output mode
        if output_mode not in ['left', 'right', 'both']:
            raise ValueError(f"output_mode must be 'left', 'right', or 'both', got: {output_mode}")
        
        # Calculate number of bins for classification mode
        if use_classification:
            self.num_bins = int((depth_range[1] - depth_range[0]) / bin_width) + 1
            print(f"🏷️  Classification mode enabled: {self.num_bins} bins, width={bin_width}mm, range={depth_range}")
        else:
            self.num_bins = None
        
        # Calculate effective input size for MLP layers
        if use_conv:
            # After 1D convolution: output_size = input_size - kernel_size + 1
            conv_output_size = input_size - conv_kernel_size + 1
            # Skip connection concatenates original input with conv output
            effective_input_size = input_size + (conv_output_size * conv_channels)
            
            # Build convolutional layer
            self.conv_layer = nn.Conv1d(
                in_channels=1,  # Input has 1 channel (single profile)
                out_channels=conv_channels,
                kernel_size=conv_kernel_size,
                padding=0  # No padding for explicit size control
            )
            
            if batch_norm:
                self.conv_bn = nn.BatchNorm1d(conv_channels)
            else:
                self.conv_bn = None
                
            self.conv_activation = self._get_activation(activation)
            
            if dropout_rate > 0:
                self.conv_dropout = nn.Dropout(dropout_rate)
            else:
                self.conv_dropout = None
                
        else:
            effective_input_size = input_size
        
        # Build shared MLP layers
        layers = []
        prev_size = effective_input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(self._get_activation(activation))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        self.shared_network = nn.Sequential(*layers)
        
        # Output heads based on mode and classification/regression
        output_size = self.num_bins if use_classification else 1
        
        if output_mode in ['left', 'both']:
            self.left_head = nn.Linear(prev_size, output_size)
        else:
            self.left_head = None
            
        if output_mode in ['right', 'both']:
            self.right_head = nn.Linear(prev_size, output_size)
        else:
            self.right_head = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }
        
        if activation.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation.lower()]
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass with optional convolutional layer and skip connections"""
        if self.use_conv:
            # Store original input for skip connection
            original_input = x  # [batch_size, input_size]
            
            # Reshape for 1D convolution: [batch_size, 1, input_size]
            x_conv = x.unsqueeze(1)
            
            # Apply convolutional layer
            conv_out = self.conv_layer(x_conv)  # [batch_size, conv_channels, conv_output_size]
            
            # Apply batch norm if enabled
            if self.conv_bn is not None:
                conv_out = self.conv_bn(conv_out)
            
            # Apply activation
            conv_out = self.conv_activation(conv_out)
            
            # Apply dropout if enabled
            if self.conv_dropout is not None:
                conv_out = self.conv_dropout(conv_out)
            
            # Flatten conv output: [batch_size, conv_channels * conv_output_size]
            conv_out_flat = conv_out.view(conv_out.size(0), -1)
            
            # Skip connection: concatenate original input with conv output
            x = torch.cat([original_input, conv_out_flat], dim=1)
        
        # Shared feature extraction through MLP
        shared_features = self.shared_network(x)
        
        # Outputs based on mode
        if self.output_mode == 'left':
            left_output = self.left_head(shared_features)
            if self.use_classification:
                output = left_output  # [batch_size, num_bins]
            else:
                output = left_output  # [batch_size, 1]
        elif self.output_mode == 'right':
            right_output = self.right_head(shared_features)
            if self.use_classification:
                output = right_output  # [batch_size, num_bins]
            else:
                output = right_output  # [batch_size, 1]
        else:  # both
            left_output = self.left_head(shared_features)
            right_output = self.right_head(shared_features)
            if self.use_classification:
                # Concatenate outputs [batch_size, 2 * num_bins]
                output = torch.cat([left_output, right_output], dim=1)
            else:
                # Concatenate outputs [batch_size, 2]
                output = torch.cat([left_output, right_output], dim=1)
        
        return output
    
    def get_architecture_info(self) -> str:
        """Get string description of architecture"""
        info = f"Dual Output MLP Architecture:\n"
        info += f"  Input size: {self.input_size}\n"
        
        if self.use_conv:
            conv_output_size = self.input_size - self.conv_kernel_size + 1
            effective_input_size = self.input_size + (conv_output_size * self.conv_channels)
            info += f"  Convolutional layer: enabled\n"
            info += f"    Channels: {self.conv_channels}\n"
            info += f"    Kernel size: {self.conv_kernel_size}\n"
            info += f"    Conv output size: {conv_output_size}\n"
            info += f"    Skip connection: original input + conv output\n"
            info += f"    Effective MLP input size: {effective_input_size}\n"
        else:
            info += f"  Convolutional layer: disabled\n"
            
        info += f"  Hidden layers: {self.hidden_layers}\n"
        
        # Task type information
        if self.use_classification:
            info += f"  Task type: Classification with binning\n"
            info += f"    Number of bins: {self.num_bins}\n"
            info += f"    Bin width: {self.bin_width} mm\n"
            info += f"    Depth range: {self.depth_range} mm\n"
        else:
            info += f"  Task type: Regression\n"
        
        # Output information based on mode
        if self.use_classification:
            if self.output_mode == 'left':
                info += f"  Output mode: Left depth only ({self.num_bins} classes)\n"
            elif self.output_mode == 'right':
                info += f"  Output mode: Right depth only ({self.num_bins} classes)\n"
            else:
                info += f"  Output mode: Both depths ({2 * self.num_bins} classes total)\n"
        else:
            if self.output_mode == 'left':
                info += f"  Output mode: Left depth only (1 output)\n"
            elif self.output_mode == 'right':
                info += f"  Output mode: Right depth only (1 output)\n"
            else:
                info += f"  Output mode: Both depths (2 outputs)\n"
            
        info += f"  Activation: {self.activation_name}\n"
        info += f"  Dropout rate: {self.dropout_rate}\n"
        info += f"  Batch normalization: {self.use_batch_norm}\n"
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info += f"  Total parameters: {total_params:,}\n"
        info += f"  Trainable parameters: {trainable_params:,}"
        
        return info

    def depths_to_bins(self, depths):
        """
        Convert continuous depth values to bin indices
        
        Args:
            depths: Array of depth values in mm (numpy array or torch tensor)
            
        Returns:
            Array of bin indices (same type as input)
        """
        if not self.use_classification:
            raise ValueError("depths_to_bins only available in classification mode")
        
        # Handle both numpy arrays and torch tensors
        if isinstance(depths, torch.Tensor):
            # PyTorch version
            depths_clamped = torch.clamp(depths, self.depth_range[0], self.depth_range[1])
            bin_indices = ((depths_clamped - self.depth_range[0]) / self.bin_width).long()
            bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        else:
            # NumPy version
            depths_clamped = np.clip(depths, self.depth_range[0], self.depth_range[1])
            bin_indices = ((depths_clamped - self.depth_range[0]) / self.bin_width).astype(int)
            bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        return bin_indices
    
    def bins_to_depths(self, bin_indices):
        """
        Convert bin indices back to continuous depth values (bin centers)
        
        Args:
            bin_indices: Array of bin indices (numpy array or torch tensor)
            
        Returns:
            Array of depth values in mm (bin centers, same type as input)
        """
        if not self.use_classification:
            raise ValueError("bins_to_depths only available in classification mode")
        
        # Handle both numpy arrays and torch tensors
        if isinstance(bin_indices, torch.Tensor):
            # PyTorch version
            depths = self.depth_range[0] + (bin_indices.float() + 0.5) * self.bin_width
        else:
            # NumPy version
            depths = self.depth_range[0] + (bin_indices + 0.5) * self.bin_width
        
        return depths
    
    def logits_to_depths(self, logits):
        """
        Convert classification logits to predicted depth values
        
        Args:
            logits: Model output logits [batch_size, num_bins] or [batch_size, 2*num_bins]
                   (numpy array or torch tensor)
            
        Returns:
            Predicted depth values [batch_size, 1] or [batch_size, 2] (numpy array)
        """
        if not self.use_classification:
            raise ValueError("logits_to_depths only available in classification mode")
        
        # Convert to numpy if it's a tensor
        if isinstance(logits, torch.Tensor):
            logits_np = logits.cpu().numpy()
        else:
            logits_np = logits
        
        if self.output_mode == 'both':
            # Split logits for left and right
            left_logits = logits_np[:, :self.num_bins]
            right_logits = logits_np[:, self.num_bins:]
            
            # Get predicted bin indices
            left_bins = np.argmax(left_logits, axis=1)
            right_bins = np.argmax(right_logits, axis=1)
            
            # Convert to depths
            left_depths = self.bins_to_depths(left_bins)
            right_depths = self.bins_to_depths(right_bins)
            
            return np.column_stack([left_depths, right_depths])
        else:
            # Single output
            predicted_bins = np.argmax(logits_np, axis=1)
            predicted_depths = self.bins_to_depths(predicted_bins)
            
            return predicted_depths.reshape(-1, 1)


class ProfileNeuralNetworkTrainer:
    """Class to handle neural network training and evaluation for profile data"""
    
    def __init__(self, model: DualOutputMLP, device: str = 'auto'):
        """
        Initialize trainer
        
        Args:
            model: Neural network model
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model = model
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_left_losses = []
        self.train_right_losses = []
        self.val_left_losses = []
        self.val_right_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              batch_size: int = 32, epochs: int = 100,
              learning_rate: float = 0.001, weight_decay: float = 1e-5,
              patience: Optional[int] = 30, min_delta: float = 0.001,
              optimizer_name: str = 'adam', scheduler_name: str = 'plateau',
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the neural network
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            patience: Early stopping patience (None to disable)
            min_delta: Minimum improvement for early stopping
            optimizer_name: Optimizer type ('adam', 'sgd', 'rmsprop')
            scheduler_name: Learning rate scheduler ('plateau', 'step', 'cosine', 'none')
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer
        optimizers = {
            'adam': optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay),
            'sgd': optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9),
            'rmsprop': optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        }
        
        if optimizer_name.lower() not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        optimizer = optimizers[optimizer_name.lower()]
        
        # Setup scheduler
        scheduler = None
        if scheduler_name.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=verbose)
        elif scheduler_name.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_name.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Loss function
        if self.model.use_classification:
            criterion = nn.CrossEntropyLoss()
            print(f"🏷️  Using CrossEntropyLoss for classification")
        else:
            criterion = nn.MSELoss()
            print(f"📏 Using MSELoss for regression")
        
        # Training loop
        self.train_losses = []
        self.val_losses = []
        self.train_left_losses = []
        self.train_right_losses = []
        self.val_left_losses = []
        self.val_right_losses = []
        patience_counter = 0
        
        # Handle infinite epochs case
        if epochs == float('inf'):
            print(f"\nStarting training until manually interrupted...")
            print(f"💡 Use Ctrl+C to stop training and save the best model")
            epoch_display = "∞"
        else:
            print(f"\nStarting training for {epochs} epochs...")
            epoch_display = str(epochs)
            
        print(f"Optimizer: {optimizer_name}, Scheduler: {scheduler_name}")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        epoch = 0
        while True:
            # Check if we should stop (for finite epochs)
            if epochs != float('inf') and epoch >= epochs:
                break
                
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_left_loss = 0.0
            train_right_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Calculate losses based on output mode and classification
                if self.model.use_classification:
                    # Convert continuous targets to bin indices for classification
                    if self.model.output_mode == 'left':
                        # Only left output, target is just left depth
                        target_left_continuous = batch_y[:, 0]
                        target_left_bins = self.model.depths_to_bins(target_left_continuous)
                        loss = criterion(outputs, target_left_bins)
                        left_loss = loss
                        right_loss = torch.tensor(0.0)  # Dummy value for tracking
                    elif self.model.output_mode == 'right':
                        # Only right output, target is just right depth
                        target_right_continuous = batch_y[:, 1]
                        target_right_bins = self.model.depths_to_bins(target_right_continuous)
                        loss = criterion(outputs, target_right_bins)
                        left_loss = torch.tensor(0.0)  # Dummy value for tracking
                        right_loss = loss
                    else:  # both
                        # Both outputs, targets are [left, right] depths converted to bins
                        target_left_bins = self.model.depths_to_bins(batch_y[:, 0])
                        target_right_bins = self.model.depths_to_bins(batch_y[:, 1])
                        
                        # For dual classification, we have separate output heads
                        outputs_left = outputs[:, :self.model.num_bins]  # First num_bins outputs for left
                        outputs_right = outputs[:, self.model.num_bins:]  # Next num_bins outputs for right
                        
                        left_loss = criterion(outputs_left, target_left_bins)
                        right_loss = criterion(outputs_right, target_right_bins)
                        loss = left_loss + right_loss
                else:
                    # Regression mode
                    if self.model.output_mode == 'left':
                        # Only left output, target is just left depth
                        target_left = batch_y[:, 0:1]  # Keep as [batch_size, 1]
                        loss = criterion(outputs, target_left)
                        left_loss = loss
                        right_loss = torch.tensor(0.0)  # Dummy value for tracking
                    elif self.model.output_mode == 'right':
                        # Only right output, target is just right depth
                        target_right = batch_y[:, 1:2]  # Keep as [batch_size, 1]
                        loss = criterion(outputs, target_right)
                        left_loss = torch.tensor(0.0)  # Dummy value for tracking
                        right_loss = loss
                    else:  # both
                        # Both outputs, target is [left, right]
                        loss = criterion(outputs, batch_y)
                        left_loss = criterion(outputs[:, 0], batch_y[:, 0])
                        right_loss = criterion(outputs[:, 1], batch_y[:, 1])
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_left_loss += left_loss.item()
                train_right_loss += right_loss.item()
            
            train_loss /= len(train_loader)
            train_left_loss /= len(train_loader)
            train_right_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_left_loss = 0.0
            val_right_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    
                    # Calculate validation losses based on output mode and classification
                    if self.model.use_classification:
                        # Convert continuous targets to bin indices for classification
                        if self.model.output_mode == 'left':
                            target_left_continuous = batch_y[:, 0]
                            target_left_bins = self.model.depths_to_bins(target_left_continuous)
                            loss = criterion(outputs, target_left_bins)
                            left_loss = loss
                            right_loss = torch.tensor(0.0)
                        elif self.model.output_mode == 'right':
                            target_right_continuous = batch_y[:, 1]
                            target_right_bins = self.model.depths_to_bins(target_right_continuous)
                            loss = criterion(outputs, target_right_bins)
                            left_loss = torch.tensor(0.0)
                            right_loss = loss
                        else:  # both
                            target_left_bins = self.model.depths_to_bins(batch_y[:, 0])
                            target_right_bins = self.model.depths_to_bins(batch_y[:, 1])
                            
                            outputs_left = outputs[:, :self.model.num_bins]
                            outputs_right = outputs[:, self.model.num_bins:]
                            
                            left_loss = criterion(outputs_left, target_left_bins)
                            right_loss = criterion(outputs_right, target_right_bins)
                            loss = left_loss + right_loss
                    else:
                        # Regression mode
                        if self.model.output_mode == 'left':
                            target_left = batch_y[:, 0:1]
                            loss = criterion(outputs, target_left)
                            left_loss = loss
                            right_loss = torch.tensor(0.0)
                        elif self.model.output_mode == 'right':
                            target_right = batch_y[:, 1:2]
                            loss = criterion(outputs, target_right)
                            left_loss = torch.tensor(0.0)
                            right_loss = loss
                        else:  # both
                            loss = criterion(outputs, batch_y)
                            left_loss = criterion(outputs[:, 0], batch_y[:, 0])
                            right_loss = criterion(outputs[:, 1], batch_y[:, 1])
                    
                    val_loss += loss.item()
                    val_left_loss += left_loss.item()
                    val_right_loss += right_loss.item()
            
            val_loss /= len(val_loader)
            val_left_loss /= len(val_loader)
            val_right_loss /= len(val_loader)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_left_losses.append(train_left_loss)
            self.train_right_losses.append(train_right_loss)
            self.val_left_losses.append(val_left_loss)
            self.val_right_losses.append(val_right_loss)
            
            # Learning rate scheduling
            if scheduler:
                if scheduler_name.lower() == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Early stopping
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if epochs == float('inf'):
                    print(f"Epoch {epoch+1:4d}/∞: "
                          f"Train Loss: {train_loss:.6f} "
                          f"(L:{train_left_loss:.6f}, R:{train_right_loss:.6f}) "
                          f"Val Loss: {val_loss:.6f} "
                          f"(L:{val_left_loss:.6f}, R:{val_right_loss:.6f})")
                else:
                    print(f"Epoch {epoch+1:4d}/{epochs}: "
                          f"Train Loss: {train_loss:.6f} "
                          f"(L:{train_left_loss:.6f}, R:{train_right_loss:.6f}) "
                          f"Val Loss: {val_loss:.6f} "
                          f"(L:{val_left_loss:.6f}, R:{val_right_loss:.6f})")
            
            # Early stopping check
            if patience and patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            # Increment epoch counter
            epoch += 1
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_left_losses': self.train_left_losses,
            'train_right_losses': self.train_right_losses,
            'val_left_losses': self.val_left_losses,
            'val_right_losses': self.val_right_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 show_samples: bool = True, num_samples: int = 10, 
                 test_metadata: List[Dict] = None, scaler_y = None, processor = None) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            y_tensor = torch.FloatTensor(y_test).to(self.device)
            predictions = self.model(X_tensor)
            
            # Convert predictions based on classification vs regression
            if self.model.use_classification:
                # For classification, convert logits to depth predictions
                y_pred = self.model.logits_to_depths(predictions)  # This already returns numpy array
            else:
                # For regression, predictions are already depths
                y_pred = predictions.cpu().numpy()
            
            y_true = y_test.copy()
            
            # Handle different output modes for denormalization
            if self.model.output_mode == 'left':
                # Predictions are [batch_size, 1], extend to [batch_size, 2] for compatibility
                y_pred_full = np.zeros((y_pred.shape[0], 2))
                if self.model.use_classification:
                    y_pred_full[:, 0] = y_pred[:, 0]  # Left predictions (already converted from classification)
                else:
                    y_pred_full[:, 0] = y_pred[:, 0]  # Left predictions
                y_pred_full[:, 1] = y_true[:, 1]  # Keep original right targets (won't be used in metrics)
                y_pred = y_pred_full
            elif self.model.output_mode == 'right':
                # Predictions are [batch_size, 1], extend to [batch_size, 2] for compatibility  
                y_pred_full = np.zeros((y_pred.shape[0], 2))
                y_pred_full[:, 0] = y_true[:, 0]  # Keep original left targets (won't be used in metrics)
                if self.model.use_classification:
                    y_pred_full[:, 1] = y_pred[:, 0]  # Right predictions (already converted from classification)
                else:
                    y_pred_full[:, 1] = y_pred[:, 0]  # Right predictions
                y_pred = y_pred_full
            # If output_mode == 'both', y_pred is already [batch_size, 2] for both regression and classification
            
            # Check if scaler is provided and fitted before using it
            scaler_fitted = False
            if hasattr(processor, 'target_left_mean') and processor.target_left_mean is not None:
                # Use custom denormalization (preferred method)
                print(f"🔄 Inverse transforming predictions and targets to original scale (using custom denormalization)...")
                y_pred = processor.denormalize_targets(y_pred)
                y_true = processor.denormalize_targets(y_true)
                print(f"✅ Values converted back to original mm units")
                scaler_fitted = True
            elif scaler_y is not None:
                try:
                    # Check if using old scaler_y (StandardScaler) 
                    _ = scaler_y.scale_
                    scaler_fitted = True
                    print(f"🔄 Inverse transforming predictions and targets to original scale (using StandardScaler)...")
                    y_pred = scaler_y.inverse_transform(y_pred)
                    y_true = scaler_y.inverse_transform(y_true)
                    print(f"✅ Values converted back to original mm units")
                except AttributeError:
                    print(f"⚠️  Scaler not fitted - using normalized values for evaluation")
                    scaler_fitted = False
            else:
                print(f"⚠️  No denormalization available - using normalized values for evaluation")
                scaler_fitted = False
            
            # Calculate metrics for overall prediction
            mse_total = mean_squared_error(y_true, y_pred)
            # Calculate metrics based on output mode
            if self.model.output_mode == 'left':
                # Only evaluate left depth metrics
                mse_left = mean_squared_error(y_true[:, 0], y_pred[:, 0])
                mae_left = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
                r2_left = r2_score(y_true[:, 0], y_pred[:, 0])
                
                # Overall metrics are just left metrics
                mse_total = mse_left
                mae_total = mae_left
                r2_total = r2_left
                
                # Right metrics are not meaningful
                mse_right = float('nan')
                mae_right = float('nan')
                r2_right = float('nan')
                
            elif self.model.output_mode == 'right':
                # Only evaluate right depth metrics
                mse_right = mean_squared_error(y_true[:, 1], y_pred[:, 1])
                mae_right = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
                r2_right = r2_score(y_true[:, 1], y_pred[:, 1])
                
                # Overall metrics are just right metrics
                mse_total = mse_right
                mae_total = mae_right
                r2_total = r2_right
                
                # Left metrics are not meaningful
                mse_left = float('nan')
                mae_left = float('nan')
                r2_left = float('nan')
                
            else:  # both
                # Calculate all metrics
                mse_total = mean_squared_error(y_true, y_pred)
                mae_total = mean_absolute_error(y_true, y_pred)
                r2_total = r2_score(y_true, y_pred)
                
                mse_left = mean_squared_error(y_true[:, 0], y_pred[:, 0])
                mae_left = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
                r2_left = r2_score(y_true[:, 0], y_pred[:, 0])
                
                mse_right = mean_squared_error(y_true[:, 1], y_pred[:, 1])
                mae_right = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
                r2_right = r2_score(y_true[:, 1], y_pred[:, 1])
            
            metrics = {
                'mse_total': mse_total,
                'mae_total': mae_total,
                'r2_total': r2_total,
                'rmse_total': np.sqrt(mse_total) if not np.isnan(mse_total) else float('nan'),
                'mse_left': mse_left,
                'mae_left': mae_left,
                'r2_left': r2_left,
                'rmse_left': np.sqrt(mse_left) if not np.isnan(mse_left) else float('nan'),
                'mse_right': mse_right,
                'mae_right': mae_right,
                'r2_right': r2_right,
                'rmse_right': np.sqrt(mse_right) if not np.isnan(mse_right) else float('nan')
            }
            
            units = "mm" if scaler_fitted else "normalized"
            print(f"Test Results ({units}) - Output Mode: {self.model.output_mode}:")
            
            if self.model.output_mode in ['left', 'both']:
                print(f"  Left    - MSE: {mse_left:.6f}, MAE: {mae_left:.6f}, R²: {r2_left:.6f}")
            if self.model.output_mode in ['right', 'both']:
                print(f"  Right   - MSE: {mse_right:.6f}, MAE: {mae_right:.6f}, R²: {r2_right:.6f}")
            if self.model.output_mode == 'both':
                print(f"  Overall - MSE: {mse_total:.6f}, MAE: {mae_total:.6f}, R²: {r2_total:.6f}")
            
            # Show sample predictions if requested
            if show_samples:
                self.print_sample_predictions(y_true, y_pred, test_metadata, num_samples, scaler_fitted)
            
            return metrics
    
    def print_sample_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               test_metadata: List[Dict] = None, num_samples: int = 10, 
                               scaler_fitted: bool = True):
        """Print sample predictions vs true values"""
        print(f"\n{'='*80}")
        print(f"SAMPLE PREDICTIONS vs TRUE VALUES")
        print(f"{'='*80}")
        
        # Get sample indices
        total_samples = len(y_true)
        sample_indices = np.linspace(0, total_samples - 1, min(num_samples, total_samples), dtype=int)
        
        print(f"Showing {len(sample_indices)} samples out of {total_samples} total test samples\n")
        
        # Units based on whether scaler was used
        units = "mm" if scaler_fitted else "norm"
        
        # Header
        print(f"{'Sample':<8} {'Hole':<15} {'Augment':<12} {'Left True':<12} {'Left Pred':<12} {'Left Error':<12} {'Right True':<12} {'Right Pred':<12} {'Right Error':<12}")
        print(f"{'#':<8} {'Name':<15} {'Type':<12} {'(' + units + ')':<12} {'(' + units + ')':<12} {'(' + units + ')':<12} {'(' + units + ')':<12} {'(' + units + ')':<12} {'(' + units + ')':<12}")
        print(f"{'-'*8} {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for i, idx in enumerate(sample_indices):
            left_true = y_true[idx, 0]
            left_pred = y_pred[idx, 0]
            right_true = y_true[idx, 1]
            right_pred = y_pred[idx, 1]
            
            left_error = abs(left_pred - left_true)
            right_error = abs(right_pred - right_true)
            
            # Get metadata if available
            hole_name = "N/A"
            augment_type = "N/A"
            if test_metadata and idx < len(test_metadata):
                metadata = test_metadata[idx]
                hole_name = metadata.get('hole_name', 'N/A')[:14]  # Truncate if too long
                augment_type = metadata.get('augmentation_type', 'original')[:11]  # Truncate if too long
            
            print(f"{i+1:<8} {hole_name:<15} {augment_type:<12} "
                  f"{left_true:<12.6f} {left_pred:<12.6f} {left_error:<12.6f} "
                  f"{right_true:<12.6f} {right_pred:<12.6f} {right_error:<12.6f}")
        
        # Calculate and print error statistics for the samples
        left_errors = np.abs(y_pred[sample_indices, 0] - y_true[sample_indices, 0])
        right_errors = np.abs(y_pred[sample_indices, 1] - y_true[sample_indices, 1])
        
        print(f"\n{'='*80}")
        print(f"SAMPLE ERROR STATISTICS")
        print(f"{'='*80}")
        print(f"Left Depth Errors ({units})  - Mean: {np.mean(left_errors):.6f}, Std: {np.std(left_errors):.6f}, Max: {np.max(left_errors):.6f}")
        print(f"Right Depth Errors ({units}) - Mean: {np.mean(right_errors):.6f}, Std: {np.std(right_errors):.6f}, Max: {np.max(right_errors):.6f}")
        
        # Show best and worst predictions
        left_best_idx = sample_indices[np.argmin(left_errors)]
        left_worst_idx = sample_indices[np.argmax(left_errors)]
        right_best_idx = sample_indices[np.argmin(right_errors)]
        right_worst_idx = sample_indices[np.argmax(right_errors)]
        
        print(f"\nBEST AND WORST PREDICTIONS:")
        print(f"Left depth  - Best:  Sample {left_best_idx} (Error: {np.min(left_errors):.6f} {units})")
        print(f"Left depth  - Worst: Sample {left_worst_idx} (Error: {np.max(left_errors):.6f} {units})")
        print(f"Right depth - Best:  Sample {right_best_idx} (Error: {np.min(right_errors):.6f} {units})")
        print(f"Right depth - Worst: Sample {right_worst_idx} (Error: {np.max(right_errors):.6f} {units})")
        
        if test_metadata:
            print(f"\nWORST PREDICTION DETAILS:")
            if left_worst_idx < len(test_metadata):
                worst_left_meta = test_metadata[left_worst_idx]
                print(f"Left worst - Hole: {worst_left_meta.get('hole_name', 'N/A')}, "
                      f"Augmentation: {worst_left_meta.get('augmentation_type', 'N/A')}")
            
            if right_worst_idx < len(test_metadata):
                worst_right_meta = test_metadata[right_worst_idx]
                print(f"Right worst - Hole: {worst_right_meta.get('hole_name', 'N/A')}, "
                      f"Augmentation: {worst_right_meta.get('augmentation_type', 'N/A')}")
    
    def predict(self, X: np.ndarray, scaler_y=None, processor=None) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input features (should be in original scale if processor is provided)
            scaler_y: Target scaler for inverse transformation
            processor: ProfileDatasetProcessor instance for feature normalization
            
        Returns:
            Predicted values (in original scale if scaler_y is provided)
        """
        self.model.eval()
        
        # Apply feature normalization if processor is provided
        X_input = X.copy()
        if processor is not None:
            try:
                X_input = processor.normalize_features_global(X_input)
                print(f"🔄 Applied global feature normalization to input data")
            except ValueError as e:
                print(f"⚠️  Could not apply feature normalization: {e}")
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_input).to(self.device)
            predictions = self.model(X_tensor)
            
            # Convert predictions based on classification vs regression
            if self.model.use_classification:
                # For classification, convert logits to depth predictions
                y_pred = self.model.logits_to_depths(predictions)  # This already returns numpy array
            else:
                # For regression, predictions are already depths
                y_pred = predictions.cpu().numpy()
            
            # Inverse transform to original scale - check custom denormalization first
            if processor is not None and hasattr(processor, 'target_left_mean') and processor.target_left_mean is not None:
                # Use custom denormalization (preferred method)
                y_pred = processor.denormalize_targets(y_pred)
            elif scaler_y is not None:
                try:
                    # Check if using old scaler_y (StandardScaler)
                    _ = scaler_y.scale_
                    y_pred = scaler_y.inverse_transform(y_pred)
                except AttributeError:
                    # Scaler not fitted, use normalized values
                    pass
            
        return y_pred
    
    def plot_training_history(self, save_path: str = None):
        """Plot training and validation loss curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Overall losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', alpha=0.8)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Overall Loss')
        ax1.set_title('Overall Training History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Left depth losses
        ax2.plot(epochs, self.train_left_losses, 'b-', label='Training Loss', alpha=0.8)
        ax2.plot(epochs, self.val_left_losses, 'r-', label='Validation Loss', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Left Depth Loss')
        ax2.set_title('Left Depth Training History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Right depth losses
        ax3.plot(epochs, self.train_right_losses, 'b-', label='Training Loss', alpha=0.8)
        ax3.plot(epochs, self.val_right_losses, 'r-', label='Validation Loss', alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Right Depth Loss')
        ax3.set_title('Right Depth Training History')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Loss comparison
        ax4.plot(epochs, self.train_left_losses, 'b-', label='Left Train', alpha=0.8)
        ax4.plot(epochs, self.train_right_losses, 'g-', label='Right Train', alpha=0.8)
        ax4.plot(epochs, self.val_left_losses, 'b--', label='Left Val', alpha=0.8)
        ax4.plot(epochs, self.val_right_losses, 'g--', label='Right Val', alpha=0.8)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Left vs Right Depth Losses')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str, metadata: Dict = None):
        """Save trained model"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_architecture': {
                'input_size': self.model.input_size,
                'hidden_layers': self.model.hidden_layers,
                'activation': self.model.activation_name,
                'dropout_rate': self.model.dropout_rate,
                'batch_norm': self.model.use_batch_norm,
                'use_conv': self.model.use_conv,
                'conv_channels': self.model.conv_channels,
                'conv_kernel_size': self.model.conv_kernel_size,
                'output_mode': self.model.output_mode,
                'use_classification': self.model.use_classification,
                'bin_width': self.model.bin_width,
                'depth_range': self.model.depth_range,
                'num_bins': getattr(self.model, 'num_bins', None)
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_left_losses': self.train_left_losses,
                'train_right_losses': self.train_right_losses,
                'val_left_losses': self.val_left_losses,
                'val_right_losses': self.val_right_losses,
                'best_val_loss': self.best_val_loss
            },
            'save_time': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'auto'):
        """Load trained model"""
        save_dict = torch.load(filepath, map_location='cpu')
        
        # Recreate model
        arch = save_dict['model_architecture']
        model = DualOutputMLP(
            input_size=arch['input_size'],
            hidden_layers=arch['hidden_layers'],
            activation=arch['activation'],
            dropout_rate=arch['dropout_rate'],
            batch_norm=arch['batch_norm'],
            use_conv=arch.get('use_conv', False),  # Backward compatibility
            conv_channels=arch.get('conv_channels', 1),
            conv_kernel_size=arch.get('conv_kernel_size', 50),
            output_mode=arch.get('output_mode', 'both'),  # Backward compatibility
            use_classification=arch.get('use_classification', False),  # Backward compatibility
            bin_width=arch.get('bin_width', 0.005),
            depth_range=arch.get('depth_range', (0.0, 1.0))
        )
        
        # Load weights
        model.load_state_dict(save_dict['model_state_dict'])
        
        # Create trainer
        trainer = cls(model, device)
        trainer.train_losses = save_dict['training_history']['train_losses']
        trainer.val_losses = save_dict['training_history']['val_losses']
        trainer.train_left_losses = save_dict['training_history']['train_left_losses']
        trainer.train_right_losses = save_dict['training_history']['train_right_losses']
        trainer.val_left_losses = save_dict['training_history']['val_left_losses']
        trainer.val_right_losses = save_dict['training_history']['val_right_losses']
        trainer.best_val_loss = save_dict['training_history']['best_val_loss']
        
        print(f"Model loaded from: {filepath}")
        return trainer, save_dict.get('metadata', {})


def main():
    """Main function with command line interface"""
    if not TORCH_AVAILABLE:
        print("PyTorch is required but not installed. Please install with:")
        print("pip install torch scikit-learn")
        return
    
    parser = argparse.ArgumentParser(description='Train neural network for profile-based countersink depth estimation')
    parser.add_argument('mode', choices=['prepare', 'train', 'evaluate'], 
                       help='Operation mode')
    parser.add_argument('--augmented-train-dir', default='augmented_dataset',
                       help='Directory containing augmented training dataset')
    parser.add_argument('--split-dataset-dir', default='split_dataset',
                       help='Directory containing validation and test datasets')
    parser.add_argument('--output-dir', default='processed_profile_datasets',
                       help='Output directory for processed datasets')
    parser.add_argument('--processed-data', default='processed_profile_datasets/processed_profile_datasets.pkl',
                       help='Path to processed dataset file')
    parser.add_argument('--model', default='profile_depth_model.pth', help='Model file path')
    parser.add_argument('--samples', type=int, default=2048, help='Number of uniform samples per profile')
    parser.add_argument('--min-angle', type=float, default=-8.0, help='Minimum angle for sampling (degrees)')
    parser.add_argument('--max-angle', type=float, default=8.0, help='Maximum angle for sampling (degrees)')
    parser.add_argument('--noise-min-neighbors', type=int, default=5, help='Minimum neighbors for noise removal (including point itself)')
    parser.add_argument('--noise-radius', type=float, default=0.05, help='Radius in mm for noise removal vicinity search')
    parser.add_argument('--disable-noise-removal', action='store_true', help='Disable noise removal preprocessing')
    parser.add_argument('--layers', nargs='+', type=int, default=[1024, 512, 256], 
                       help='Hidden layer sizes')
    parser.add_argument('--activation', default='relu', help='Activation function')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=70, help='Early stopping patience (epochs)')
    parser.add_argument('--min-delta', type=float, default=0.001, help='Minimum improvement for early stopping')
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--run-until-interrupted', action='store_true', help='Run indefinitely until manually interrupted (Ctrl+C)')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization strength')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'rmsprop'], help='Optimizer type')
    parser.add_argument('--scheduler', default='plateau', choices=['plateau', 'step', 'cosine', 'none'], help='Learning rate scheduler')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--no-batch-norm', action='store_true', help='Disable batch normalization')
    parser.add_argument('--use-conv', action='store_true', help='Use convolutional layer as first layer with skip connections')
    parser.add_argument('--conv-channels', type=int, default=1, help='Number of output channels for convolutional layer')
    parser.add_argument('--conv-kernel-size', type=int, default=50, help='Kernel size (width) for convolutional layer')
    parser.add_argument('--use-classification', action='store_true', help='Use classification with binning instead of regression')
    parser.add_argument('--bin-width', type=float, default=0.005, help='Width of each depth bin for classification (mm)')
    parser.add_argument('--depth-range', type=float, nargs=2, default=[0.0, 1.0], help='Min and max depth range for binning (mm) - ignored in classification mode where range is calculated from training data')
    parser.add_argument('--output-mode', default='both', choices=['left', 'right', 'both'], help='Which outputs to train: left depth only, right depth only, or both')
    parser.add_argument('--show-samples', action='store_true', help='Show sample predictions vs true values in evaluate mode')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of sample predictions to show')
    parser.add_argument('--results-file', default='evaluation_results.json', help='Output file for evaluation results (JSON format)')
    
    args = parser.parse_args()
    
    print("PROFILE NEURAL NETWORK TRAINING")
    print("=" * 50)
    
    if args.mode == 'prepare':
        # Prepare datasets
        processor = ProfileDatasetProcessor(
            augmented_train_dir=args.augmented_train_dir,
            split_dataset_dir=args.split_dataset_dir,
            uniform_samples=args.samples,
            min_angle=args.min_angle,
            max_angle=args.max_angle,
            noise_min_neighbors=args.noise_min_neighbors,
            noise_radius=args.noise_radius,
            enable_noise_removal=not args.disable_noise_removal
        )
        
        processor.load_datasets()
        processor.prepare_datasets()
        processor.save_processed_datasets(args.output_dir)
        
    elif args.mode == 'train':
        # Load processed datasets
        processor = ProfileDatasetProcessor.load_processed_datasets(args.processed_data)
        
        # Calculate depth range from training data if using classification
        if args.use_classification:
            y_train = processor.y_train
            min_depth = float(np.min(y_train))
            max_depth = float(np.max(y_train))
            # Add small buffer to ensure all training samples are within range
            depth_buffer = (max_depth - min_depth) * 0.05  # 5% buffer
            calculated_depth_range = (min_depth - depth_buffer, max_depth + depth_buffer)
            print(f"📊 Calculated depth range from training data: {calculated_depth_range}")
            print(f"   Training depth range: [{min_depth:.3f}, {max_depth:.3f}] mm")
            depth_range_to_use = calculated_depth_range
        else:
            depth_range_to_use = tuple(args.depth_range)
        
        # Create model
        model = DualOutputMLP(
            input_size=processor.X_train.shape[1],
            hidden_layers=args.layers,
            activation=args.activation,
            dropout_rate=args.dropout,
            batch_norm=not args.no_batch_norm,
            use_conv=args.use_conv,
            conv_channels=args.conv_channels,
            conv_kernel_size=args.conv_kernel_size,
            output_mode=args.output_mode,
            use_classification=args.use_classification,
            bin_width=args.bin_width,
            depth_range=depth_range_to_use
        )
        
        print(model.get_architecture_info())
        
        # Train model
        trainer = ProfileNeuralNetworkTrainer(model)
        
        # Configure early stopping and epochs
        if args.run_until_interrupted:
            # Run indefinitely until Ctrl+C
            patience = None
            epochs = float('inf')  # Infinite epochs
            print("🔄 Running until manually interrupted (Ctrl+C)...")
            print("💡 Model will save best state continuously and can be interrupted safely")
        else:
            patience = None if args.no_early_stopping else args.patience
            epochs = args.epochs
        
        try:
            history = trainer.train(
                processor.X_train, processor.y_train, 
                processor.X_val, processor.y_val,
                epochs=epochs, 
                batch_size=args.batch_size, 
                learning_rate=args.lr, 
                patience=patience,
                min_delta=args.min_delta, 
                weight_decay=args.weight_decay,
                optimizer_name=args.optimizer, 
                scheduler_name=args.scheduler
            )
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user (Ctrl+C)")
            print("💾 Best model state has been preserved")
            # Training history is still available for plotting
            history = {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'train_left_losses': trainer.train_left_losses,
                'train_right_losses': trainer.train_right_losses,
                'val_left_losses': trainer.val_left_losses,
                'val_right_losses': trainer.val_right_losses,
                'best_val_loss': trainer.best_val_loss
            }
        
        # Evaluate on test set
        if processor.X_test is not None:
            metrics = trainer.evaluate(
                processor.X_test, 
                processor.y_test, 
                show_samples=False,  # Don't show samples during training
                test_metadata=processor.test_metadata,
                scaler_y=processor.scaler_y,
                processor=processor
            )
        else:
            metrics = {}
        
        # Save model
        trainer.save_model(args.model, metadata={'test_metrics': metrics})
        trainer.plot_training_history('profile_training_history.png')
        
    elif args.mode == 'evaluate':
        # Load model and evaluate
        trainer, metadata = ProfileNeuralNetworkTrainer.load_model(args.model)
        
        # Load test data
        processor = ProfileDatasetProcessor.load_processed_datasets(args.processed_data)
        
        if processor.X_test is not None:
            print(f"\n🔍 Evaluating model on test dataset...")
            print(f"📊 Test set size: {len(processor.X_test)} samples")
            if args.show_samples:
                print(f"📋 Will show {args.num_samples} sample predictions")
            
            metrics = trainer.evaluate(
                processor.X_test, 
                processor.y_test, 
                show_samples=args.show_samples,
                num_samples=args.num_samples,
                test_metadata=processor.test_metadata,
                scaler_y=processor.scaler_y,
                processor=processor
            )
            
            # Generate predictions for all test samples
            predictions = trainer.predict(processor.X_test, scaler_y=processor.scaler_y, processor=processor)
            
            # Create results in the same format as results.json
            results = {}
            for i, (pred, meta) in enumerate(zip(predictions, processor.test_metadata)):
                hole_name = meta['hole_name']
                
                # Convert predictions to the format expected
                left_depth = float(pred[0])
                right_depth = float(pred[1])
                
                results[hole_name] = {
                    "left_total_depth": left_depth,
                    "right_total_depth": right_depth
                }
            
            # Save results to JSON file
            results_file = args.results_file
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Evaluation results saved to: {results_file}")
            print(f"📊 Results format matches target results.json structure")
            print(f"🎯 Generated predictions for {len(results)} holes")
            
            # Print additional analysis
            if args.show_samples:
                print(f"\n📈 FULL DATASET STATISTICS:")
                print(f"Test set contains {len(processor.X_test)} samples")
                if processor.test_metadata:
                    # Count augmentation types
                    augment_counts = {}
                    for meta in processor.test_metadata:
                        aug_type = meta.get('augmentation_type', 'original')
                        augment_counts[aug_type] = augment_counts.get(aug_type, 0) + 1
                    
                    print(f"Augmentation breakdown:")
                    for aug_type, count in augment_counts.items():
                        print(f"  {aug_type}: {count} samples ({count/len(processor.test_metadata)*100:.1f}%)")
        else:
            print("No test data available for evaluation")


if __name__ == "__main__":
    main()