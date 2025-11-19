#!/usr/bin/env python3
"""
PCD File Prediction Script for Countersink Depth Estimation
-----------------------------------------------------------
Loads a trained neural network model and predicts countersink depths from a single PCD file.
Uses the same preprocessing pipeline as training (uniform angle sampling, noise removal, normalization).

Usage:
    python predict_from_pcd.py input.pcd --model model.pth --processed-data processed_datasets.pkl
    
Features:
- Loads PCD file and applies same preprocessing as training
- Uses normalization parameters from training dataset
- Makes predictions for left and right countersink depths
- Handles both classification and regression models
- Supports different analysis methods (per-sample vs global normalization)

Author: GitHub Copilot
Date: November 19, 2025
"""

import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
from create_profile_dataset import ProfileDatasetCreator

# Deep learning imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Please install: pip install torch")
    TORCH_AVAILABLE = False

# Import the training modules
try:
    from train_profile_neural_network import (
        ProfileDatasetProcessor, 
        ProfileNeuralNetworkTrainer, 
        DualOutputMLP
    )
    # Import convert_pcd from parent directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from convert_pcd import PCDConverter
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure train_profile_neural_network.py is in the same directory and convert_pcd.py is in the parent directory")
    sys.exit(1)


class PCDPredictor:
    """Class to handle PCD file prediction using trained neural network models"""
    
    def __init__(self, model_path: str, processed_data_path: str, device: str = 'auto'):
        """
        Initialize PCD predictor
        
        Args:
            model_path: Path to trained model (.pth file)
            processed_data_path: Path to processed dataset (.pkl file) containing normalization parameters
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required but not installed. Please install with: pip install torch")
        
        self.model_path = Path(model_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Validate file paths
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not self.processed_data_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {processed_data_path}")
        
        # Load model
        print(f"🔄 Loading trained model from: {self.model_path}")
        self.trainer, self.model_metadata = ProfileNeuralNetworkTrainer.load_model(
            str(self.model_path)        )
        
        # Load preprocessing parameters
        print(f"🔄 Loading preprocessing parameters from: {self.processed_data_path}")
        self.processor = ProfileDatasetProcessor.load_processed_datasets(str(self.processed_data_path))
        
        # Initialize PCD converter
        self.pcd_converter = PCDConverter()
        
        print(f"✅ PCD Predictor initialized successfully")
        print(f"   Model architecture: {self.trainer.model.get_architecture_info().split('Dual Output MLP Architecture:')[1].split('Hidden layers:')[0].strip()}")
        print(f"   Output mode: {self.trainer.model.output_mode}")
        print(f"   Task type: {'Classification' if self.trainer.model.use_classification else 'Regression'}")
        
        # Display normalization info
        if hasattr(self.processor, 'per_sample_normalization') and self.processor.per_sample_normalization:
            print(f"   Feature normalization: Per-sample (subtract each sample's mean)")
        else:
            print(f"   Feature normalization: Global (subtract global mean, divide by global std)")
    
    def load_and_preprocess_pcd(self, pcd_file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load and preprocess a single PCD file using the same pipeline as training
        
        Args:
            pcd_file_path: Path to PCD file
            
        Returns:
            Tuple of (processed_features, metadata)
        """
        pcd_path = Path(pcd_file_path)
        if not pcd_path.exists():
            raise FileNotFoundError(f"PCD file not found: {pcd_file_path}")
        
        print(f"\n🔄 Loading and preprocessing: {pcd_path.name}")
        
        # Step 1: Load raw PCD file
        print(f"  📁 Loading raw PCD file...")
        x_raw, y_raw, z_raw, header_lines = self.pcd_converter.load_pcd_file(str(pcd_path))
        print(f"     Loaded {len(x_raw):,} points")
        
        # Step 2: Apply coordinate transformations (same as training)
        x_converted = x_raw * 1000.0
        y_converted = y_raw * 1000.0
        z_converted = z_raw * 1000.0
        print(f"     Converted {len(x_converted):,} points (no transformation applied)")
        
        # Step 3: Apply uniform angle sampling (use processor's method)
        print(f"  📐 Applying uniform angle sampling...")
        uniform_z = self.processor.uniform_angle_sampling(x_converted, y_converted, z_converted)
        
        # Step 4: Handle NaN values (use processor's method)
        print(f"  🔧 Handling NaN values...")
        processed_features = self.processor.handle_nan_values(uniform_z, method='interpolate')
        
        # Create metadata
        metadata = {
            'original_file': str(pcd_path),
            'filename': pcd_path.name,
            'stem': pcd_path.stem,
            'original_points': len(x_raw),
            'converted_points': len(x_converted),
            'uniform_samples': len(processed_features),
            'nan_count': np.sum(np.isnan(uniform_z)),
            'preprocessing_method': 'uniform_angle_sampling + interpolation'
        }
        
        print(f"  ✅ Preprocessing complete")
        print(f"     Original points: {metadata['original_points']:,}")
        print(f"     Uniform samples: {metadata['uniform_samples']:,}")
        print(f"     NaN values handled: {metadata['nan_count']:,}")
        
        return processed_features.reshape(1, -1), metadata  # Add batch dimension

    
    def predict(self, pcd_file_path: str, verbose: bool = True) -> Dict:
        """
        Make predictions on a single PCD file
        
        Args:
            pcd_file_path: Path to PCD file
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess the PCD file
        features, metadata = self.load_and_preprocess_pcd(pcd_file_path)
        
        # Apply feature normalization (same as training)
        if verbose:
            print(f"\n🔄 Applying feature normalization...")
        
        # features_normalized = self.processor.normalize_features_global(features)
        
        # if hasattr(self.processor, 'per_sample_normalization') and self.processor.per_sample_normalization:
        #     if verbose:
        #         print(f"   Applied per-sample normalization (subtracted sample mean)")
        # else:
        #     if verbose:
        #         print(f"   Applied global normalization (global mean/std)")
        
        # Make prediction
        if verbose:
            print(f"\n🎯 Making prediction...")
        
        predictions = self.trainer.predict(features, processor=self.processor)  # Already normalized
        print("PRedictions: ", predictions)
        # Extract predictions based on output mode
        if self.trainer.model.output_mode == 'left':
            left_depth = float(predictions[0, 0])
            right_depth = None
        elif self.trainer.model.output_mode == 'right':
            left_depth = None
            right_depth = float(predictions[0, 0])
        else:  # both
            left_depth = float(predictions[0, 0])
            right_depth = float(predictions[0, 1])
        
        # Compile results
        results = {
            'predictions': {
                'left_depth_mm': left_depth,
                'right_depth_mm': right_depth
            },
            'model_info': {
                'output_mode': self.trainer.model.output_mode,
                'task_type': 'classification' if self.trainer.model.use_classification else 'regression',
                'model_path': str(self.model_path),
                'processed_data_path': str(self.processed_data_path)
            },
            'preprocessing': {
                'uniform_samples': self.processor.uniform_samples,
                'angle_range': [self.processor.min_angle, self.processor.max_angle],
                'noise_removal_enabled': self.processor.enable_noise_removal,
                'normalization_type': 'per_sample' if hasattr(self.processor, 'per_sample_normalization') and self.processor.per_sample_normalization else 'global'
            },
            'file_info': metadata
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"File: {metadata['filename']}")
            print(f"Original points: {metadata['original_points']:,}")
            print(f"Processed samples: {metadata['uniform_samples']:,}")
            
            if left_depth is not None:
                print(f"Left countersink depth:  {left_depth:.4f} mm")
            if right_depth is not None:
                print(f"Right countersink depth: {right_depth:.4f} mm")
            
            print(f"\nModel: {self.trainer.model.output_mode} output, {results['model_info']['task_type']}")
            print(f"Normalization: {results['preprocessing']['normalization_type']}")
            print(f"{'='*60}")
        
        return results


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Predict countersink depths from PCD file using trained neural network')
    parser.add_argument('pcd_file', help='Path to PCD file to analyze')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--processed-data', required=True, 
                       help='Path to processed dataset file (.pkl) containing normalization parameters')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--output', help='Output JSON file to save results')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    print("PCD COUNTERSINK DEPTH PREDICTION")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = PCDPredictor(
            model_path=args.model,
            processed_data_path=args.processed_data,
            device=args.device
        )
        
        # Make prediction
        results = predictor.predict(args.pcd_file, verbose=not args.quiet)
        
        # Save results if requested
        if args.output:
            import json
            output_path = Path(args.output)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            
            json_results = convert_numpy_types(results)
            
            with open(output_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n💾 Results saved to: {output_path}")
        
        # Return success
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())