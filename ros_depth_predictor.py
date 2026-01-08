#!/usr/bin/env python3

"""
ROS Wrapper for Neural Network Depth Prediction
-----------------------------------------------
This ROS node provides a service that accepts a PointCloud2 message and returns 
predicted countersink depths using a trained neural network model.

Features:
- Provides ROS service for depth prediction
- Uses trained neural network model from predict_from_pcd.py
- Converts ROS PointCloud2 to required format
- Returns left and right predicted depths
- Handles different model output modes (left, right, both)

Usage:
    rosrun state_machine ros_depth_predictor.py
    
    # With custom model paths
    rosrun state_machine ros_depth_predictor.py --model custom_model.pth --processed-data custom_data.pkl
    
Services:
    /depth_prediction (state_machine/DepthPrediction) - Predict depths from point cloud
    
Parameters:
    ~model_path: Path to trained model file (default: profile_depth_model.pth)
    ~processed_data_path: Path to processed dataset file (default: processed_profile_datasets/processed_profile_datasets.pkl)  
    ~device: Device for inference ('auto', 'cpu', 'cuda') (default: auto)
"""

from __future__ import print_function

import rospy
import numpy as np
import os
import sys
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from state_machine.srv import DepthPrediction, DepthPredictionResponse

# Add the 2d_estimation directory to the path so we can import prediction modules
script_dir = os.path.dirname(os.path.abspath(__file__))
_2d_estimation_dir = os.path.join(script_dir, '2d_estimation')
sys.path.append(_2d_estimation_dir)

# Import prediction functionality
try:
    from predict_from_pcd import PCDPredictor
except ImportError as e:
    rospy.logerr("Error importing prediction modules: {}".format(e))
    rospy.logerr("Please ensure predict_from_pcd.py and related modules are in the 2d_estimation directory")
    sys.exit(1)


class ROSDepthPredictor:
    """ROS wrapper for neural network depth prediction"""
    
    def __init__(self):
        """Initialize the ROS node and prediction components"""
        rospy.init_node('ros_depth_predictor', anonymous=True)
        
        # Get parameters from ROS parameter server or use defaults
        script_dir = os.path.dirname(os.path.abspath(__file__))
        eslam_dir = os.path.join(script_dir, '2d_estimation')
        
        default_model = os.path.join(eslam_dir, 'profile_depth_model.pth')
        default_processed_data = os.path.join(eslam_dir, 'processed_profile_datasets', 'processed_profile_datasets.pkl')
        
        self.model_path = rospy.get_param('~model_path', default_model)
        self.processed_data_path = rospy.get_param('~processed_data_path', default_processed_data)
        self.device = rospy.get_param('~device', 'auto')
        
        rospy.loginfo("Initializing ROS Depth Predictor...")
        rospy.loginfo("Model path: {}".format(self.model_path))
        rospy.loginfo("Processed data path: {}".format(self.processed_data_path))
        rospy.loginfo("Device: {}".format(self.device))
        
        # Validate file paths
        if not os.path.exists(self.model_path):
            rospy.logerr("Model file not found: {}".format(self.model_path))
            sys.exit(1)
        if not os.path.exists(self.processed_data_path):
            rospy.logerr("Processed data file not found: {}".format(self.processed_data_path))
            sys.exit(1)
        
        # Initialize the predictor
        try:
            self.predictor = PCDPredictor(
                model_path=self.model_path,
                processed_data_path=self.processed_data_path,
                device=self.device
            )
            rospy.loginfo("Neural network model loaded successfully")
        except Exception as e:
            rospy.logerr("Failed to initialize predictor: {}".format(e))
            sys.exit(1)
        
        # ROS Service
        self.prediction_service = rospy.Service('/depth_prediction', DepthPrediction, self.service_callback)
        
        rospy.loginfo("ROS Depth Predictor initialized")
        rospy.loginfo("Service available at: /depth_prediction")
        rospy.loginfo("Model output mode: {}".format(self.predictor.trainer.model.output_mode))
        rospy.loginfo("Task type: {}".format('classification' if self.predictor.trainer.model.use_classification else 'regression'))
        
    def pointcloud2_to_numpy(self, msg):
        """
        Convert sensor_msgs/PointCloud2 to numpy array
        
        Args:
            msg (sensor_msgs/PointCloud2): Input point cloud message
            
        Returns:
            np.ndarray: Point cloud as numpy array (N x 3) with x, y, z coordinates
        """
        # Extract points from PointCloud2 message
        points_list = []
        
        # Use sensor_msgs point_cloud2 module for conversion
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        
        if len(points_list) == 0:
            return np.array([]).reshape(0, 3)
            
        points = np.array(points_list)
        
        # Filter out infinite values
        finite_mask = np.isfinite(points).all(axis=1)
        points_clean = points[finite_mask]
        
        if len(points_clean) < len(points):
            removed = len(points) - len(points_clean)
            rospy.logwarn("Removed {} points with infinite values".format(removed))
        
        return points_clean
    
    def predict_from_pointcloud2(self, msg):
        """
        Predict depths from PointCloud2 message
        
        Args:
            msg (sensor_msgs/PointCloud2): Input point cloud message
            
        Returns:
            dict: Prediction results
        """
        # Convert to numpy
        points = self.pointcloud2_to_numpy(msg)
        
        if len(points) == 0:
            raise ValueError("Empty point cloud")
        
        # Create a temporary PCD-like data structure that the predictor can work with
        # We'll simulate the preprocessing directly with numpy arrays
        x_raw, y_raw, z_raw = points[:, 0], points[:, 1], points[:, 2]
        
        rospy.logdebug("Converted PointCloud2 to numpy: {} points".format(len(points)))
        
        # Apply coordinate transformations (same as training - but already in correct units)
        # The neural network expects data in mm, so we convert from meters if needed
        # Check if the data is in meters (typical range) vs mm
        if np.max(np.abs(points)) < 10:  # Likely in meters, convert to mm
            x_converted = x_raw * 1000.0
            y_converted = y_raw * 1000.0  
            z_converted = z_raw * 1000.0
            rospy.logdebug("Converted from meters to millimeters")
        else:  # Already in mm
            x_converted = x_raw
            y_converted = y_raw
            z_converted = z_raw
            rospy.logdebug("Data already in millimeters")
        
        # Apply uniform angle sampling (use processor's method)
        rospy.logdebug("Applying uniform angle sampling...")
        uniform_z = self.predictor.processor.uniform_angle_sampling(x_converted, y_converted, z_converted)
        
        # Handle NaN values (use processor's method)
        rospy.logdebug("Handling NaN values...")
        processed_features = self.predictor.processor.handle_nan_values(uniform_z, method='interpolate')
        
        # Add batch dimension and make prediction
        features = processed_features.reshape(1, -1)
        rospy.logdebug("Making prediction with {} features...".format(features.shape[1]))
        
        predictions = self.predictor.trainer.predict(features, processor=self.predictor.processor)
        
        # Extract predictions based on output mode
        if self.predictor.trainer.model.output_mode == 'left':
            left_depth = float(predictions[0, 0])
            right_depth = None
            left_valid = True
            right_valid = False
        elif self.predictor.trainer.model.output_mode == 'right':
            left_depth = None
            right_depth = float(predictions[0, 0])
            left_valid = False
            right_valid = True
        else:  # both
            left_depth = float(predictions[0, 0])
            right_depth = float(predictions[0, 1])
            left_valid = True
            right_valid = True
        
        # Create metadata
        metadata = {
            'original_points': len(points),
            'uniform_samples': len(processed_features),
            'nan_count': np.sum(np.isnan(uniform_z)),
            'left_depth': left_depth,
            'right_depth': right_depth,
            'left_valid': left_valid,
            'right_valid': right_valid,
            'model_output_mode': self.predictor.trainer.model.output_mode,
            'task_type': 'classification' if self.predictor.trainer.model.use_classification else 'regression',
            'normalization_type': 'per_sample' if hasattr(self.predictor.processor, 'per_sample_normalization') and self.predictor.processor.per_sample_normalization else 'global'
        }
        
        return metadata
    
    def service_callback(self, req):
        """
        Service callback function for DepthPrediction requests
        
        Args:
            req (DepthPredictionRequest): Service request containing point cloud
            
        Returns:
            DepthPredictionResponse: Prediction results
        """
        rospy.loginfo("Received DepthPrediction service request with {} points".format(
            req.pointcloud.width * req.pointcloud.height))
        
        response = DepthPredictionResponse()
        
        try:
            # Make prediction
            results = self.predict_from_pointcloud2(req.pointcloud)
            
            # Fill response
            response.success = True
            response.error_message = ""
            
            response.left_valid = results['left_valid']
            response.left_depth = results['left_depth'] if results['left_depth'] is not None else 0.0
            
            response.right_valid = results['right_valid']
            response.right_depth = results['right_depth'] if results['right_depth'] is not None else 0.0
            
            response.model_output_mode = results['model_output_mode']
            response.task_type = results['task_type']
            response.normalization_type = results['normalization_type']
            
            response.original_points = results['original_points']
            response.processed_samples = results['uniform_samples']
            response.nan_count = results['nan_count']
            
            rospy.loginfo("Prediction completed successfully:")
            if results['left_valid']:
                rospy.loginfo("  Left depth: {:.4f} mm".format(results['left_depth']))
            if results['right_valid']:
                rospy.loginfo("  Right depth: {:.4f} mm".format(results['right_depth']))
            
            return response
                
        except Exception as e:
            rospy.logerr("Error processing prediction request: {}".format(e))
            import traceback
            rospy.logerr(traceback.format_exc())
            
            # Fill error response
            response.success = False
            response.error_message = str(e)
            response.left_valid = False
            response.right_valid = False
            response.left_depth = 0.0
            response.right_depth = 0.0
            response.model_output_mode = ""
            response.task_type = ""
            response.normalization_type = ""
            response.original_points = 0
            response.processed_samples = 0
            response.nan_count = 0
            
            return response
    
    def run(self):
        """Run the ROS node"""
        rospy.loginfo("ROS Depth Predictor running... Press Ctrl+C to stop")
        rospy.spin()


def main():
    """Main function"""
    try:
        predictor = ROSDepthPredictor()
        predictor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Depth Predictor shutdown")
    except Exception as e:
        rospy.logerr("Fatal error: {}".format(e))
        import traceback
        rospy.logerr(traceback.format_exc())


if __name__ == '__main__':
    main()
