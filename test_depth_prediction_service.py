#!/usr/bin/env python3

"""
Test Script for Neural Network Depth Prediction Service
-------------------------------------------------------
This script tests the DepthPrediction ROS service by:
1. Loading a specific PCD file (Hole-1194_0.0mm.pcd)
2. Converting it to PointCloud2 format
3. Calling the depth prediction service
4. Displaying the predicted results

Usage:
    # Test with default PCD file (Hole-1194_0.0mm.pcd)
    rosrun state_machine test_depth_prediction_service.py
    
    # Test with custom PCD file
    rosrun state_machine test_depth_prediction_service.py /path/to/file.pcd
    
    # Test with custom service name
    rosrun state_machine test_depth_prediction_service.py --service /custom_service
    
Prerequisites:
    1. roscore must be running
    2. ros_depth_predictor.py node must be running
    
Example full workflow:
    # Terminal 1: Start roscore
    roscore
    
    # Terminal 2: Start the prediction node
    rosrun state_machine ros_depth_predictor.py
    
    # Terminal 3: Run this test
    rosrun state_machine test_depth_prediction_service.py
"""

from __future__ import print_function

import rospy
import sys
import os
import argparse
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from state_machine.srv import DepthPrediction, DepthPredictionRequest


class DepthPredictionServiceTester:
    """Test client for DepthPrediction service"""
    
    def __init__(self, service_name='/depth_prediction'):
        """
        Initialize the service tester
        
        Args:
            service_name (str): Name of the service to call
        """
        self.service_name = service_name
        
        # Initialize ROS node
        rospy.init_node('depth_prediction_service_tester', anonymous=True)
        
        rospy.loginfo("DepthPrediction Service Tester initialized")
        rospy.loginfo("Will call service: {}".format(service_name))
    
    def load_pcd_file(self, filename):
        """
        Load PCD file and return point coordinates
        
        Args:
            filename (str): Path to PCD file
            
        Returns:
            np.ndarray: Point cloud as numpy array (N x 3)
        """
        if not os.path.exists(filename):
            raise FileNotFoundError("PCD file not found: {}".format(filename))
        
        rospy.loginfo("Loading PCD file: {}".format(filename))
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Find the start of data
        data_start = 0
        points_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('POINTS'):
                points_count = int(line.split()[1])
            elif line.startswith('DATA'):
                if 'ascii' not in line.lower():
                    raise ValueError("Only ASCII PCD files are supported")
                data_start = i + 1
                break
        
        if data_start == 0:
            raise ValueError("Could not find DATA section in PCD file")
        
        rospy.loginfo("Expected points: {:,}".format(points_count))
        
        # Extract point data
        points = []
        invalid_points = 0
        
        for line in lines[data_start:]:
            line = line.strip()
            if line:
                try:
                    values = line.split()
                    if len(values) >= 3:
                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                        
                        # Check for invalid values
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                            points.append([x, y, z])
                        else:
                            invalid_points += 1
                except ValueError:
                    invalid_points += 1
        
        if len(points) == 0:
            raise ValueError("No valid points found in PCD file")
        
        points_array = np.array(points, dtype=np.float32)
        
        if invalid_points > 0:
            rospy.logwarn("Filtered out {} invalid points".format(invalid_points))
        
        rospy.loginfo("Loaded {} valid points".format(len(points_array)))
        
        return points_array
    
    def create_pointcloud2_msg(self, points, frame_id='base_link'):
        """
        Create a sensor_msgs/PointCloud2 message from point array
        
        Args:
            points (np.ndarray): Point cloud as numpy array (N x 3)
            frame_id (str): Frame ID for the point cloud
            
        Returns:
            sensor_msgs/PointCloud2: Point cloud message
        """
        # Create header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        
        # Define point fields (x, y, z as float32)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        
        # Create PointCloud2 message
        msg = pc2.create_cloud(header, fields, points)
        
        return msg
    
    def call_service(self, pointcloud_msg):
        """
        Call the DepthPrediction service
        
        Args:
            pointcloud_msg (sensor_msgs/PointCloud2): Point cloud to predict
            
        Returns:
            DepthPredictionResponse: Service response
        """
        rospy.loginfo("Waiting for service: {}".format(self.service_name))
        
        try:
            # Wait for service to be available
            rospy.wait_for_service(self.service_name, timeout=30.0)
            
            # Create service proxy
            predict_service = rospy.ServiceProxy(self.service_name, DepthPrediction)
            
            # Create request
            request = DepthPredictionRequest()
            request.pointcloud = pointcloud_msg
            
            rospy.loginfo("Calling service with {} points...".format(
                pointcloud_msg.width * pointcloud_msg.height))
            
            # Call service
            response = predict_service(request)
            
            rospy.loginfo("Service call completed")
            return response
            
        except rospy.ROSException as e:
            rospy.logerr("Service call failed: {}".format(e))
            raise
    
    def print_results(self, response):
        """
        Print service response in a formatted way
        
        Args:
            response (DepthPredictionResponse): Service response
        """
        print("\n" + "=" * 70)
        print("NEURAL NETWORK DEPTH PREDICTION RESULTS")
        print("=" * 70)
        
        print("Overall Success: {}".format("✅ YES" if response.success else "❌ NO"))
        if response.error_message:
            print("Error Message: {}".format(response.error_message))
        
        print("Original Points: {:,}".format(response.original_points))
        print("Processed Samples: {:,}".format(response.processed_samples))
        if response.nan_count > 0:
            print("NaN Values Handled: {:,}".format(response.nan_count))
        print("")
        
        # Model information
        print("MODEL INFORMATION:")
        print("-" * 20)
        print("  Output Mode: {}".format(response.model_output_mode))
        print("  Task Type: {}".format(response.task_type))
        print("  Normalization: {}".format(response.normalization_type))
        print("")
        
        # Prediction results
        print("PREDICTED DEPTHS:")
        print("-" * 20)
        
        if response.left_valid:
            print("  🔵 Left Depth: {:.4f} mm".format(response.left_depth))
        else:
            print("  🔵 Left Depth: Not predicted by this model")
        
        if response.right_valid:
            print("  🟠 Right Depth: {:.4f} mm".format(response.right_depth))
        else:
            print("  🟠 Right Depth: Not predicted by this model")
        
        # Summary
        if response.left_valid and response.right_valid:
            avg_depth = (response.left_depth + response.right_depth) / 2
            depth_diff = abs(response.left_depth - response.right_depth)
            print("")
            print("SUMMARY (Both Sides):")
            print("-" * 20)
            print("  📊 Average Depth: {:.4f} mm".format(avg_depth))
            print("  📐 Depth Difference: {:.4f} mm".format(depth_diff))
        elif response.left_valid or response.right_valid:
            single_depth = response.left_depth if response.left_valid else response.right_depth
            side = "Left" if response.left_valid else "Right"
            print("")
            print("SUMMARY (Single Side):")
            print("-" * 20)
            print("  📊 {} Depth: {:.4f} mm".format(side, single_depth))
        
        print("=" * 70)
    
    def test_with_file(self, pcd_file):
        """
        Test the service with a specific PCD file
        
        Args:
            pcd_file (str): Path to PCD file
        """
        try:
            # Load PCD file
            points = self.load_pcd_file(pcd_file)
            
            # Convert to PointCloud2 message
            pointcloud_msg = self.create_pointcloud2_msg(points)
            
            # Call service
            response = self.call_service(pointcloud_msg)
            
            # Print results
            self.print_results(response)
            
            return response
            
        except Exception as e:
            rospy.logerr("Test failed: {}".format(e))
            import traceback
            rospy.logerr(traceback.format_exc())
            return None


def main():
    """Main function with command line argument parsing"""
    # Default PCD file path as specified in the request
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_pcd = os.path.join(script_dir, '2d_estimation', 'single_profile_test_pcd', 'Hole-1194_0.0mm.pcd')
    
    parser = argparse.ArgumentParser(description='Test DepthPrediction ROS service')
    parser.add_argument('pcd_file', nargs='?', default=default_pcd,
                       help='Path to PCD file (default: Hole-1194_0.0mm.pcd)')
    parser.add_argument('--service', default='/depth_prediction',
                       help='Service name to call (default: /depth_prediction)')
    
    args = parser.parse_args()
    
    # Validate PCD file
    if not os.path.exists(args.pcd_file):
        print("❌ PCD file not found: {}".format(args.pcd_file))
        print("\nTrying to find available test PCD files...")
        
        # Look for test files in various locations
        search_dirs = [
            os.path.join(script_dir, '2d_estimation', 'single_profile_test_pcd'),
            os.path.join(script_dir, '2d_estimation'),
            script_dir
        ]
        
        found_pcds = []
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                pcd_files = [f for f in os.listdir(search_dir) if f.endswith('.pcd')]
                for f in pcd_files:
                    full_path = os.path.join(search_dir, f)
                    found_pcds.append(full_path)
        
        if found_pcds:
            print("\nAvailable PCD files:")
            for pcd in sorted(found_pcds)[:10]:  # Show first 10
                print("  - {}".format(pcd))
            if len(found_pcds) > 10:
                print("  ... and {} more".format(len(found_pcds) - 10))
        
        return 1
    
    try:
        print("🚀 Starting DepthPrediction Service Test")
        print("📁 PCD File: {}".format(args.pcd_file))
        print("🔧 Service: {}".format(args.service))
        print("")
        
        # Create tester
        tester = DepthPredictionServiceTester(args.service)
        
        # Run test
        response = tester.test_with_file(args.pcd_file)
        
        if response is not None:
            print("\n✅ Test completed successfully!")
            
            # Return appropriate exit code
            if response.success:
                return 0  # Success
            else:
                return 2  # Prediction failed
        else:
            print("\n❌ Test failed!")
            return 3
        
    except rospy.ROSInterruptException:
        print("\n🛑 Test interrupted by user")
        return 0
    except Exception as e:
        print("\n❌ Fatal error: {}".format(e))
        import traceback
        traceback.print_exc()
        return 4


if __name__ == '__main__':
    sys.exit(main())
