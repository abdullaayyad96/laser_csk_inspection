#!/usr/bin/env python3

"""
Test Script for CountersinkAnalysis Service
-------------------------------------------
This script tests the CountersinkAnalysis ROS service by:
1. Loading a PCD file and converting it to PointCloud2
2. Calling the service with the point cloud
3. Displaying the results

Usage:
    # Test with default PCD file
    rosrun state_machine test_countersink_service.py
    
    # Test with custom PCD file
    rosrun state_machine test_countersink_service.py /path/to/file.pcd
    
    # Test with custom service name
    rosrun state_machine test_countersink_service.py --service /custom_service
    
Prerequisites:
    1. roscore must be running
    2. ros_single_pcd_analysis.py node must be running
    
Example full workflow:
    # Terminal 1: Start roscore
    roscore
    
    # Terminal 2: Start the analysis node
    rosrun state_machine ros_single_pcd_analysis.py
    
    # Terminal 3: Run this test
    rosrun state_machine test_countersink_service.py
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
from state_machine.srv import CountersinkAnalysis, CountersinkAnalysisRequest


class CountersinkServiceTester:
    """Test client for CountersinkAnalysis service"""
    
    def __init__(self, service_name='/countersink_analysis'):
        """
        Initialize the service tester
        
        Args:
            service_name (str): Name of the service to call
        """
        self.service_name = service_name
        
        # Initialize ROS node
        rospy.init_node('countersink_service_tester', anonymous=True)
        
        rospy.loginfo("CountersinkAnalysis Service Tester initialized")
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
        Call the CountersinkAnalysis service
        
        Args:
            pointcloud_msg (sensor_msgs/PointCloud2): Point cloud to analyze
            
        Returns:
            CountersinkAnalysisResponse: Service response
        """
        rospy.loginfo("Waiting for service: {}".format(self.service_name))
        
        try:
            # Wait for service to be available
            rospy.wait_for_service(self.service_name, timeout=30.0)
            
            # Create service proxy
            analyze_service = rospy.ServiceProxy(self.service_name, CountersinkAnalysis)
            
            # Create request
            request = CountersinkAnalysisRequest()
            request.pointcloud = pointcloud_msg
            
            rospy.loginfo("Calling service with {} points...".format(
                pointcloud_msg.width * pointcloud_msg.height))
            
            # Call service
            response = analyze_service(request)
            
            rospy.loginfo("Service call completed")
            return response
            
        except rospy.ROSException as e:
            rospy.logerr("Service call failed: {}".format(e))
            raise
    
    def print_results(self, response):
        """
        Print service response in a formatted way
        
        Args:
            response (CountersinkAnalysisResponse): Service response
        """
        print("\n" + "=" * 70)
        print("COUNTERSINK ANALYSIS SERVICE RESULTS")
        print("=" * 70)
        
        print("Overall Success: {}".format("✅ YES" if response.success else "❌ NO"))
        if response.error_message:
            print("Error Message: {}".format(response.error_message))
        print("Total Points: {:,}".format(response.total_points))
        print("")
        
        # Left side results
        print("LEFT SIDE ANALYSIS:")
        print("-" * 20)
        if response.left_success:
            print("  ✅ Success: YES")
            print("  📏 Depth: {:.3f} mm".format(response.left_depth))
            print("  📊 RMSE: {:.3f} mm".format(response.left_rmse))
            print("  🔢 Hole Points: {:,}".format(response.left_hole_points))
        else:
            print("  ❌ Success: NO")
            print("  Error: {}".format(response.left_error))
        print("")
        
        # Right side results
        print("RIGHT SIDE ANALYSIS:")
        print("-" * 20)
        if response.right_success:
            print("  ✅ Success: YES")
            print("  📏 Depth: {:.3f} mm".format(response.right_depth))
            print("  📊 RMSE: {:.3f} mm".format(response.right_rmse))
            print("  🔢 Hole Points: {:,}".format(response.right_hole_points))
        else:
            print("  ❌ Success: NO")
            print("  Error: {}".format(response.right_error))
        print("")
        
        # Summary results
        print("SUMMARY:")
        print("-" * 20)
        if response.success:
            print("  📊 Average Depth: {:.3f} mm".format(response.average_depth))
            print("  📐 Depth Difference: {:.3f} mm".format(response.depth_difference))
            print("  🎯 Consistency: {}".format(response.consistency_rating))
            print("  🔵 Left Depth: {:.3f} mm".format(response.left_depth))
            print("  🟠 Right Depth: {:.3f} mm".format(response.right_depth))
        else:
            print("  ⚠️  Analysis incomplete - one or both sides failed")
            if response.left_success and not response.right_success:
                print("  🔵 Left Depth (only): {:.3f} mm".format(response.left_depth))
            elif response.right_success and not response.left_success:
                print("  🟠 Right Depth (only): {:.3f} mm".format(response.right_depth))
        
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
    parser = argparse.ArgumentParser(description='Test CountersinkAnalysis ROS service')
    parser.add_argument('pcd_file', nargs='?', 
                       default='/home/abdulla/codes/stata_mobile_robot_upgraded_ws/src/strata_robotic_machining/state_machine/scripts/csk_laser/november_11_cont_scan/Hole-1168_3d.pcd',
                       help='Path to PCD file (default: Hole-1168_3d.pcd)')
    parser.add_argument('--service', default='/countersink_analysis',
                       help='Service name to call (default: /countersink_analysis)')
    
    args = parser.parse_args()
    
    # Validate PCD file
    if not os.path.exists(args.pcd_file):
        print("❌ PCD file not found: {}".format(args.pcd_file))
        print("\nAvailable PCD files in november_11_cont_scan/:")
        scan_dir = "/home/abdulla/codes/stata_mobile_robot_upgraded_ws/src/strata_robotic_machining/state_machine/scripts/csk_laser/november_11_cont_scan/"
        if os.path.exists(scan_dir):
            pcd_files = [f for f in os.listdir(scan_dir) if f.endswith('.pcd')]
            for f in sorted(pcd_files):
                print("  - {}".format(os.path.join(scan_dir, f)))
        return 1
    
    try:
        print("🚀 Starting CountersinkAnalysis Service Test")
        print("📁 PCD File: {}".format(args.pcd_file))
        print("🔧 Service: {}".format(args.service))
        print("")
        
        # Create tester
        tester = CountersinkServiceTester(args.service)
        
        # Run test
        response = tester.test_with_file(args.pcd_file)
        
        if response is not None:
            print("\n✅ Test completed successfully!")
            
            # Return appropriate exit code
            if response.success:
                return 0  # Both sides successful
            elif response.left_success or response.right_success:
                return 1  # Partial success
            else:
                return 2  # Complete failure
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
