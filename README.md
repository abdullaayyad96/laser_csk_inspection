# Countersink Depth Estimation (CSK Laser)

This directory contains tools and algorithms for automated countersink depth estimation using laser scanning technology. The system supports both **full 3D scans** and **single 2D profile scans** for accurate countersink depth measurements in robotic machining applications.

## Overview

Countersink depth estimation is critical for quality control in aerospace manufacturing, particularly for nutplate installation holes. This package provides:

- **3D Point Cloud Analysis**: Full 3D reconstruction and analysis of countersink geometry
- **2D Profile Analysis**: Efficient single-scan depth estimation using neural networks
- **Offline Testing Tools**: For algorithm development and validation
- **Online ROS Integration**: Real-time depth estimation services

---

## 1. Estimation from Full 3D Scan

Full 3D scanning provides the most comprehensive analysis of countersink geometry by capturing the complete surface topology.

### A. Offline Methods (Algorithm Testing)

These tools are designed for testing algorithms on existing point cloud data:

#### 1. Create Dummy 3D Point Cloud
**File**: `countersink_3d_scanner.py`

Generate synthetic countersink point clouds for testing purposes.

```bash
python countersink_3d_scanner.py
```

Creates realistic 3D point clouds with configurable countersink parameters for algorithm validation.

#### 2. Point Cloud Transformations
**File**: `convert_pcd.py`

*Note: Not needed for simulated data, but required for real sensor data*

Real laser scanner data often requires coordinate transformations:
- Convert from meters to millimeters
- Flip Z-axis orientation (analysis code assumes countersink extrudes inwards in the negative z axis)

```bash
python convert_pcd.py <input.pcd> <output.pcd> --meters-to-mm --flip-z
```

The convert_pcd.py utility provides:
- Unit scaling (meters to millimeters)
- Axis flipping 


#### 3. Split Point Cloud (Left/Right Countersinks)
**File**: `test_pcd_loading_and_splitting.py`

Split point clouds containing two countersink holes (common in nutplate installation).

```bash
python test_pcd_loading_and_splitting.py <input.pcd>
```

Automatically detects and separates left and right countersink regions for individual analysis.

#### 4. Single Countersink Estimation
**File**: `countersink_depth_estimator.py`

Estimate depth from a point cloud containing a single countersink hole.

```bash
python countersink_depth_estimator.py <input.pcd>
```

*Note: You may need to run coordinate transformations (step 2) first to ensure proper units and orientation.*

#### 5. Dual Countersink Estimation
**File**: `single_pcd_analysis.py`

Analyze point clouds with both left and right countersink holes. Includes all necessary coordinate conversions.

```bash
python single_pcd_analysis.py <input.pcd>
```

This script automatically:
- Applies coordinate transformations
- Splits the point cloud
- Estimates depth for both countersinks
- Provides comprehensive analysis results

#### 6. Batch Processing
**File**: `batch_countersink_analysis.py`

Process multiple point clouds in a directory for large-scale analysis.

```bash
python batch_countersink_analysis.py --input-dir /path/to/pcd/files --output-dir /path/to/results
```

Efficiently processes entire datasets with parallel processing and detailed reporting.

### B. Online Implementation (ROS Integration)

#### Real-time ROS Service
**File**: `ros_single_pcd_analysis.py`

Provides a ROS service for real-time countersink depth estimation.

```bash
# Launch the ROS service
rosrun state_machine ros_single_pcd_analysis.py
```

The service subscribes to point cloud topics and provides depth estimation results via ROS services.

#### Testing the ROS Service
**File**: `test_countersink_service.py`

Test the ROS-based countersink estimation service.

```bash
python test_countersink_service.py
```

Sends test point clouds to the ROS service and validates the responses.

---

## 2. Estimation from Single 2D Scan

2D profile-based estimation offers faster processing using neural network models trained on profile data.

### Offline Methods

#### Test Pretrained Model
**File**: `predict_from_pcd.py`

Use a pretrained neural network to estimate countersink depth from a single 2D scan.

```bash
python predict_from_pcd.py <input.pcd> --model <model.pth> --processed-data <processed_datasets.pkl>
```

**Example**:
```bash
python predict_from_pcd.py sample_scan.pcd --model profile_depth_model.pth --processed-data processed_profile_datasets/processed_profile_datasets.pkl
```

**Options**:
- `--model`: Path to trained neural network model (.pth file)
- `--processed-data`: Path to preprocessing parameters (.pkl file)
- `--output`: Save results to JSON file
- `--quiet`: Suppress detailed output

### Online Implementation (ROS Integration)

#### Real-time 2D Analysis Service
**File**: `ros_depth_predictor.py`

ROS service for real-time 2D profile-based depth prediction.

```bash
rosrun state_machine ros_depth_predictor.py
```

#### Testing the 2D ROS Service
**File**: `test_depth_prediction_service.py`

Test the 2D neural network-based depth prediction service.

```bash
python test_depth_prediction_service.py
```

---

## 3. Neural Network Training (2D Estimation)

The `2d_estimation/` subdirectory contains tools for training and managing neural network models:

### Dataset Preparation
- `extract_pcd_files.py`: Extract PCD files from ZIP archives
- `extract_profile_json_files.py`: Extract profile JSON files from archives
- `create_profile_dataset.py`: Create training datasets from profiles and depth labels

### Model Training
- `train_profile_neural_network.py`: Train neural networks for depth estimation
- `augment_data.py`: Data augmentation for improved model generalization
- `split_dataset.py`: Split datasets into train/validation/test sets

### Visualization
- `visualize_profile_dataset.py`: Visualize training datasets
- `visualize_processed_samples.py`: Inspect preprocessed training samples

---

## 4. Utilities

### Point Cloud Conversion
**File**: `convert_pcd.py`

Convert between different point cloud formats and coordinate systems.

### Point Cloud Publishing
**File**: `pcd_publisher.py`

ROS node for publishing point cloud data from files.

### Analysis Utilities
**File**: `point_cloud_utils.py`

Common utilities for point cloud processing and analysis.

---

## 5. Getting Started

### Prerequisites

```bash
# Python dependencies
pip install numpy matplotlib scipy scikit-learn torch open3d

# ROS dependencies
sudo apt-get install ros-noetic-pcl-ros ros-noetic-sensor-msgs
```

### Quick Start - 3D Analysis

```bash
# 1. Test with synthetic data
python countersink_3d_scanner.py

# 2. Analyze a single point cloud
python single_pcd_analysis.py example.pcd

# 3. Process multiple files
python batch_countersink_analysis.py --input-dir data/ --output-dir results/
```

### Quick Start - 2D Analysis

```bash
# 1. Test pretrained model
cd 2d_estimation/
python predict_from_pcd.py sample.pcd --model profile_depth_model.pth

# 2. Start ROS service
rosrun state_machine ros_depth_predictor.py
```

---

## 6. File Organization

```
csk_laser/
├── README.md                           # This file
├── convert_pcd.py                      # Point cloud format conversion
├── countersink_3d_scanner.py           # Synthetic 3D data generation
├── countersink_depth_estimator.py      # Single countersink analysis
├── single_pcd_analysis.py              # Dual countersink analysis
├── batch_countersink_analysis.py       # Batch processing
├── ros_single_pcd_analysis.py          # ROS service (3D)
├── ros_depth_predictor.py              # ROS service (2D)
├── test_countersink_service.py         # Test 3D ROS service
├── test_depth_prediction_service.py    # Test 2D ROS service
├── test_pcd_loading_and_splitting.py   # Point cloud splitting test
├── point_cloud_utils.py                # Utility functions
├── pcd_publisher.py                    # ROS point cloud publisher
└── 2d_estimation/                      # Neural network training
    ├── predict_from_pcd.py              # 2D depth prediction
    ├── train_profile_neural_network.py  # Model training
    ├── create_profile_dataset.py        # Dataset creation
    ├── augment_data.py                  # Data augmentation
    └── ...                              # Additional training tools
```

---

## 7. ROS Services and API

### Traditional 3D Analysis Service

**Service**: `/countersink_analysis` (CountersinkAnalysis.srv)

```bash
# Request
sensor_msgs/PointCloud2 pointcloud
---
# Response
bool success
string error_message
bool left_success
float32 left_depth
float32 left_rmse
int32 left_hole_points
string left_error
bool right_success
float32 right_depth
float32 right_rmse
int32 right_hole_points
string right_error
float32 average_depth
float32 depth_difference
string consistency_rating
int32 total_points
```

### 2D Neural Network Service

**Service**: `/depth_prediction` (DepthPrediction.srv)

```bash
# Request
sensor_msgs/PointCloud2 pointcloud
---
# Response
bool success
string error_message
bool left_valid
float32 left_depth
bool right_valid
float32 right_depth
string model_output_mode
string task_type
string normalization_type
int32 original_points
int32 processed_samples
int32 nan_count
```

### Usage Examples

**Traditional 3D Analysis:**
```bash
# Terminal 1: Start roscore
roscore

# Terminal 2: Start analysis service
rosrun state_machine ros_single_pcd_analysis.py

# Terminal 3: Test the service
python test_countersink_service.py
```

**2D Neural Network Prediction:**
```bash
# Terminal 1: Start roscore  
roscore

# Terminal 2: Start prediction service
rosrun state_machine ros_depth_predictor.py

# Terminal 3: Test the service
python test_depth_prediction_service.py
```

---

## 8. Performance Notes

- **3D Analysis**: More accurate but computationally intensive (~1-2 seconds per scan)
- **2D Analysis**: Faster processing using neural networks (~0.1-0.2 seconds per scan)
- **Accuracy**: 3D methods typically achieve ±0.1mm accuracy, 2D methods achieve ±0.15mm accuracy
- **Use Cases**: 
  - 3D for critical quality control and detailed geometry analysis
  - 2D for real-time production monitoring and high-throughput inspection

---

## 9. Troubleshooting

### Common Issues

1. **Coordinate System Problems**: Ensure Z-axis orientation is correct (countersink should extend outward)
2. **Unit Conversion**: Verify data is in millimeters, not meters
3. **Missing Dependencies**: Install all required Python packages and ROS dependencies
4. **Model Files**: Ensure neural network models and preprocessing parameters are available

### ROS Debugging

```bash
# Check available services
rosservice list | grep -E "(countersink|depth_prediction)"

# Check node status
rosnode list

# View logs
rosrun rqt_console rqt_console
```

### Service Testing

```bash
# Test traditional analysis service
rosservice call /countersink_analysis "pointcloud: {}"

# Test neural network service
rosservice call /depth_prediction "pointcloud: {}"
```

---

## 10. Contact

For questions or issues, contact:
- **Author**: Abdulla Ayyad <abdullaayyad96@gmail.com>
- **Repository**: AdvancedResearchInnovationCenter/strata_robotic_machining

---

*Last Updated: January 8, 2026*
