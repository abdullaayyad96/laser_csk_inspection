#!/usr/bin/env python3

"""
ROS Wrapper for Single PCD Countersink Analysis
-----------------------------------------------
This ROS node receives a PointCloud2 message via a topic and processes it 
using the single_pcd_analysis pipeline to estimate countersink depth.

Features:
- Subscribes to PointCloud2 messages
- Converts ROS PointCloud2 to numpy arrays 
- Processes point cloud through single_pcd_analysis pipeline
- Publishes results as custom message and prints to console
- Saves temporary PCD files for processing

Usage:
    rosrun state_machine ros_single_pcd_analysis.py
    
Required ROS topics:
    /input_pointcloud (sensor_msgs/PointCloud2) - Input point cloud
    
Published topics:
    /countersink_analysis_result - Analysis results
    
Parameters:
    ~output_folder: Output folder for intermediate files (default: /tmp/ros_pcd_analysis)
    ~save_intermediate: Save intermediate PCD files (default: True)
    ~generate_plots: Generate visualization plots (default: False)
    ~analysis_method: 'separate' or 'global' (default: 'global')
"""

from __future__ import print_function

import rospy
import numpy as np
import os
import sys
import tempfile
from pathlib import Path
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, Header
import sensor_msgs.point_cloud2 as pc2
from state_machine.srv import CountersinkAnalysis, CountersinkAnalysisResponse

# Add the csk_laser directory to the path so we can import our analysis modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import our analysis functionality
try:
    from single_pcd_analysis import SinglePCDAnalyzer
    from countersink_depth_estimator import CountersinkDepthEstimator
    from test_pcd_loading_and_splitting import PCDTester
    from convert_pcd import PCDConverter
except ImportError as e:
    rospy.logerr("Error importing analysis modules: {}".format(e))
    rospy.logerr("Please ensure all required scripts are in the csk_laser directory")
    sys.exit(1)


class ROSCountersinkAnalyzer:
    """ROS wrapper for countersink analysis"""
    
    def __init__(self):
        """Initialize the ROS node and analysis components"""
        rospy.init_node('ros_countersink_analyzer', anonymous=True)
        
        # Get parameters from ROS parameter server
        self.output_folder = rospy.get_param('~output_folder', '/tmp/ros_pcd_analysis')
        self.save_intermediate = rospy.get_param('~save_intermediate', True)
        self.generate_plots = rospy.get_param('~generate_plots', True)
        self.analysis_method = rospy.get_param('~analysis_method', 'global')
        
        # Create output folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize estimator parameters (can be set via ROS params)
        self.estimator_params = {
            'expected_csk_angle_deg': rospy.get_param('~csk_angle', 100.0),
            'expected_inner_radius': rospy.get_param('~inner_radius', 1.2446),
            'optimizer': rospy.get_param('~optimizer', 'slsqp'),
            'noise_neighbors': rospy.get_param('~noise_neighbors', 100),
            'noise_radius': rospy.get_param('~noise_radius', 0.5),
            'outlier_method': rospy.get_param('~outlier_method', 'percentile'),
            'outlier_threshold': rospy.get_param('~outlier_threshold', 0.02),
            'random_seed': rospy.get_param('~random_seed', 42)
        }
        
        # Initialize the analyzer
        self.analyzer = SinglePCDAnalyzer(
            estimator_params=self.estimator_params,
            save_intermediate=self.save_intermediate,
            generate_plots=self.generate_plots,
            analysis_method=self.analysis_method
        )
        
        # ROS Publishers and Subscribers
        self.result_pub = rospy.Publisher('/countersink_analysis_result', String, queue_size=1)
        self.pointcloud_sub = rospy.Subscriber('/input_pointcloud', PointCloud2, self.pointcloud_callback, queue_size=1)
        
        # ROS Service
        self.analysis_service = rospy.Service('/countersink_analysis', CountersinkAnalysis, self.service_callback)
        
        rospy.loginfo("ROS Countersink Analyzer initialized")
        rospy.loginfo("Subscribed to: /input_pointcloud")
        rospy.loginfo("Publishing to: /countersink_analysis_result")
        rospy.loginfo("Service available at: /countersink_analysis")
        rospy.loginfo("Output folder: {}".format(self.output_folder))
        rospy.loginfo("Analysis method: {}".format(self.analysis_method))
        
    def pointcloud_callback(self, msg):
        """
        Callback function for PointCloud2 messages
        
        Args:
            msg (sensor_msgs/PointCloud2): Input point cloud message
        """
        rospy.loginfo("Received PointCloud2 with {} points".format(msg.width * msg.height))
        
        try:
            # Convert PointCloud2 to numpy arrays
            points = self.pointcloud2_to_numpy(msg)
            
            if len(points) == 0:
                rospy.logwarn("Empty point cloud received")
                return
                
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            rospy.loginfo("Converted to numpy arrays: {} points".format(len(x)))
            
            # Save as temporary PCD file for processing
            temp_pcd_file = self.save_temp_pcd(x, y, z, msg.header.stamp)
            
            # Process the point cloud
            results = self.process_pointcloud(temp_pcd_file)
            
            # Publish and print results
            self.publish_results(results, msg.header)
            
            # Clean up temporary file
            if os.path.exists(temp_pcd_file):
                os.remove(temp_pcd_file)
                
        except Exception as e:
            rospy.logerr("Error processing point cloud: {}".format(e))
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def service_callback(self, req):
        """
        Service callback function for CountersinkAnalysis requests
        
        Args:
            req (CountersinkAnalysisRequest): Service request containing point cloud
            
        Returns:
            CountersinkAnalysisResponse: Analysis results
        """
        rospy.loginfo("Received CountersinkAnalysis service request with {} points".format(
            req.pointcloud.width * req.pointcloud.height))
        
        response = CountersinkAnalysisResponse()
        
        try:
            # Convert PointCloud2 to numpy arrays
            points = self.pointcloud2_to_numpy(req.pointcloud)
            
            if len(points) == 0:
                rospy.logwarn("Empty point cloud received in service request")
                response.success = False
                response.error_message = "Empty point cloud received"
                response.left_success = False
                response.right_success = False
                response.left_error = "Empty point cloud"
                response.right_error = "Empty point cloud"
                return response
                
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            rospy.loginfo("Converted to numpy arrays: {} points".format(len(x)))
            
            # Save as temporary PCD file for processing
            temp_pcd_file = self.save_temp_pcd(x, y, z, req.pointcloud.header.stamp)
            
            # Process the point cloud
            results = self.process_pointcloud(temp_pcd_file)
            
            # Fill response based on results
            self.fill_service_response(response, results)
            
            # Clean up temporary file
            if os.path.exists(temp_pcd_file):
                os.remove(temp_pcd_file)
            
            rospy.loginfo("Service request processed successfully")
            return response
                
        except Exception as e:
            rospy.logerr("Error processing service request: {}".format(e))
            import traceback
            rospy.logerr(traceback.format_exc())
            
            # Fill error response
            response.success = False
            response.error_message = str(e)
            response.left_success = False
            response.right_success = False
            response.left_error = str(e)
            response.right_error = str(e)
            
            return response
    
    def fill_service_response(self, response, results):
        """
        Fill the service response based on analysis results
        
        Args:
            response (CountersinkAnalysisResponse): Response to fill
            results (dict): Analysis results
        """
        left_results = results['left_results']
        right_results = results['right_results']
        bilateral_analysis = results['bilateral_analysis']
        
        # Overall success
        response.success = bilateral_analysis['both_successful']
        response.total_points = results['original_points']
        
        # Left side results
        response.left_success = left_results['success']
        if left_results['success']:
            response.left_depth = float(left_results['estimated_total_depth'])
            response.left_rmse = float(left_results['fit_rmse'])
            response.left_hole_points = int(left_results['hole_points'])
            response.left_error = ""
        else:
            response.left_depth = 0.0
            response.left_rmse = 0.0
            response.left_hole_points = 0
            response.left_error = left_results.get('error_message', 'Analysis failed')
        
        # Right side results
        response.right_success = right_results['success']
        if right_results['success']:
            response.right_depth = float(right_results['estimated_total_depth'])
            response.right_rmse = float(right_results['fit_rmse'])
            response.right_hole_points = int(right_results['hole_points'])
            response.right_error = ""
        else:
            response.right_depth = 0.0
            response.right_rmse = 0.0
            response.right_hole_points = 0
            response.right_error = right_results.get('error_message', 'Analysis failed')
        
        # Summary results (if both successful)
        if bilateral_analysis['both_successful']:
            response.average_depth = float(bilateral_analysis['average_depth_mm'])
            response.depth_difference = float(bilateral_analysis['depth_difference_mm'])
            response.consistency_rating = bilateral_analysis['depth_consistency']
            response.error_message = ""
        else:
            response.average_depth = 0.0
            response.depth_difference = 0.0
            response.consistency_rating = "failed"
            response.error_message = "One or both sides failed analysis"
    
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
    
    def save_temp_pcd(self, x, y, z, timestamp):
        """
        Save point cloud as temporary PCD file
        
        Args:
            x, y, z (np.ndarray): Point coordinates
            timestamp (rospy.Time): Timestamp for filename
            
        Returns:
            str: Path to temporary PCD file
        """
        # Create unique filename with timestamp
        timestamp_str = "{}.{}".format(timestamp.secs, timestamp.nsecs)
        temp_filename = "ros_pointcloud_{}.pcd".format(timestamp_str)
        temp_filepath = os.path.join(self.output_folder, temp_filename)
        
        # Write PCD file
        num_points = len(x)
        with open(temp_filepath, 'w') as f:
            # Write PCD header
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write("WIDTH {}\n".format(num_points))
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write("POINTS {}\n".format(num_points))
            f.write("DATA ascii\n")
            
            # Write point data
            for i in range(num_points):
                f.write("{:.6f} {:.6f} {:.6f}\n".format(x[i], y[i], z[i]))
        
        rospy.logdebug("Saved temporary PCD file: {}".format(temp_filepath))
        return temp_filepath
    
    def process_pointcloud(self, pcd_file):
        """
        Process the point cloud using single_pcd_analysis pipeline
        
        Args:
            pcd_file (str): Path to PCD file
            
        Returns:
            dict: Analysis results
        """
        rospy.loginfo("Processing point cloud: {}".format(os.path.basename(pcd_file)))
        
        try:
            # Run the analysis pipeline
            results = self.analyzer.process_pcd_file(pcd_file, self.output_folder)
            return results
            
        except Exception as e:
            rospy.logerr("Analysis failed: {}".format(e))
            raise
    
    def publish_results(self, results, header):
        """
        Publish and print analysis results
        
        Args:
            results (dict): Analysis results
            header (std_msgs/Header): Original message header
        """
        # Extract key results
        left_results = results['left_results']
        right_results = results['right_results']
        bilateral_analysis = results['bilateral_analysis']
        
        # Create summary message
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("COUNTERSINK DEPTH ANALYSIS RESULTS")
        summary_lines.append("=" * 60)
        summary_lines.append("Timestamp: {}".format(header.stamp))
        summary_lines.append("Frame ID: {}".format(header.frame_id))
        summary_lines.append("Original Points: {:,}".format(results['original_points']))
        summary_lines.append("")
        
        # Left side results
        summary_lines.append("LEFT SIDE:")
        if left_results['success']:
            summary_lines.append("  ✅ Depth: {:.3f} mm".format(left_results['estimated_total_depth']))
            summary_lines.append("  📊 RMSE: {:.3f} mm".format(left_results['fit_rmse']))
            summary_lines.append("  🔢 Hole Points: {:,}".format(left_results['hole_points']))
        else:
            summary_lines.append("  ❌ Failed: {}".format(left_results.get('error_message', 'Unknown error')))
        
        summary_lines.append("")
        
        # Right side results
        summary_lines.append("RIGHT SIDE:")
        if right_results['success']:
            summary_lines.append("  ✅ Depth: {:.3f} mm".format(right_results['estimated_total_depth']))
            summary_lines.append("  📊 RMSE: {:.3f} mm".format(right_results['fit_rmse']))
            summary_lines.append("  🔢 Hole Points: {:,}".format(right_results['hole_points']))
        else:
            summary_lines.append("  ❌ Failed: {}".format(right_results.get('error_message', 'Unknown error')))
        
        summary_lines.append("")
        
        # Summary if both successful
        if bilateral_analysis['both_successful']:
            avg_depth = bilateral_analysis['average_depth_mm']
            depth_diff = bilateral_analysis['depth_difference_mm']
            consistency = bilateral_analysis['depth_consistency']
            
            summary_lines.append("SUMMARY:")
            summary_lines.append("  Average Depth: {:.3f} mm".format(avg_depth))
            summary_lines.append("  Depth Difference: {:.3f} mm".format(depth_diff))
            summary_lines.append("  Consistency: {}".format(consistency))
            summary_lines.append("  Left:  {:.3f} mm".format(left_results['estimated_total_depth']))
            summary_lines.append("  Right: {:.3f} mm".format(right_results['estimated_total_depth']))
        else:
            summary_lines.append("SUMMARY:")
            summary_lines.append("  Analysis incomplete - one or both sides failed")
        
        summary_lines.append("=" * 60)
        
        # Join all lines
        summary_text = "\n".join(summary_lines)
        
        # Print to console
        rospy.loginfo("Analysis completed:")
        for line in summary_lines:
            rospy.loginfo(line)
        
        # Publish as ROS message
        result_msg = String()
        result_msg.data = summary_text
        self.result_pub.publish(result_msg)
        
        rospy.loginfo("Results published to /countersink_analysis_result")
    
    def run(self):
        """Run the ROS node"""
        rospy.loginfo("ROS Countersink Analyzer running... Press Ctrl+C to stop")
        rospy.spin()


def main():
    """Main function"""
    try:
        analyzer = ROSCountersinkAnalyzer()
        analyzer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Countersink Analyzer shutdown")
    except Exception as e:
        rospy.logerr("Fatal error: {}".format(e))
        import traceback
        rospy.logerr(traceback.format_exc())


if __name__ == '__main__':
    main()
