#!/usr/bin/env python3

"""
PCD File Publisher for ROS
--------------------------
This script loads a PCD file and publishes it as a sensor_msgs/PointCloud2 message 
on a ROS topic. Designed to work with the ros_single_pcd_analysis.py node.

Features:
- Loads PCD files with robust error handling
- Converts to sensor_msgs/PointCloud2 format
- Publishes at configurable rate
- Supports one-shot or continuous publishing
- Filters out infinite/NaN values

Usage:
    # Publish once
    rosrun state_machine pcd_publisher.py /path/to/file.pcd
    
    # Publish continuously at 1 Hz
    rosrun state_machine pcd_publisher.py /path/to/file.pcd --rate 1.0 --continuous
    
    # Publish to custom topic
    rosrun state_machine pcd_publisher.py /path/to/file.pcd --topic /my_pointcloud
    
Published topics:
    /input_pointcloud (sensor_msgs/PointCloud2) - Point cloud data
    
Parameters:
    --topic: Output topic name (default: /input_pointcloud)
    --rate: Publishing rate in Hz (default: 0.5)
    --frame-id: Frame ID for point cloud (default: 'base_link')
    --continuous: Publish continuously (default: publish once)
    --delay: Initial delay before publishing in seconds (default: 2.0)
"""

from __future__ import print_function

import rospy
import numpy as np
import os
import sys
import argparse
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2


class PCDPublisher:
    """Publisher for PCD files to ROS PointCloud2 topics"""
    
    def __init__(self, pcd_file, topic_name='/input_pointcloud', frame_id='base_link'):
        """
        Initialize the PCD publisher
        
        Args:
            pcd_file (str): Path to PCD file
            topic_name (str): ROS topic name to publish on
            frame_id (str): Frame ID for the point cloud
        """
        self.pcd_file = pcd_file
        self.topic_name = topic_name
        self.frame_id = frame_id
        
        # Load the PCD file
        self.points = self.load_pcd_file(pcd_file)
        
        # Initialize ROS
        rospy.init_node('pcd_publisher', anonymous=True)
        
        # Create publisher
        self.pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        
        rospy.loginfo("PCD Publisher initialized")
        rospy.loginfo("File: {}".format(pcd_file))
        rospy.loginfo("Points: {:,}".format(len(self.points)))
        rospy.loginfo("Publishing to: {}".format(topic_name))
        rospy.loginfo("Frame ID: {}".format(frame_id))
    
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
    
    def create_pointcloud2_msg(self):
        """
        Create a sensor_msgs/PointCloud2 message from the loaded points
        
        Returns:
            sensor_msgs/PointCloud2: Point cloud message
        """
        # Create header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        
        # Define point fields (x, y, z as float32)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        
        # Create PointCloud2 message
        msg = pc2.create_cloud(header, fields, self.points)
        
        return msg
    
    def publish_once(self, delay=2.0):
        """
        Publish the point cloud once
        
        Args:
            delay (float): Delay before publishing in seconds
        """
        rospy.loginfo("Waiting {:.1f} seconds before publishing...".format(delay))
        rospy.sleep(delay)
        
        if rospy.is_shutdown():
            return
            
        msg = self.create_pointcloud2_msg()
        
        rospy.loginfo("Publishing point cloud with {:,} points...".format(len(self.points)))
        self.pub.publish(msg)
        
        rospy.loginfo("Point cloud published successfully")
        rospy.loginfo("Message details:")
        rospy.loginfo("  Timestamp: {}".format(msg.header.stamp))
        rospy.loginfo("  Frame ID: {}".format(msg.header.frame_id))
        rospy.loginfo("  Width: {}".format(msg.width))
        rospy.loginfo("  Height: {}".format(msg.height))
        rospy.loginfo("  Point step: {}".format(msg.point_step))
        rospy.loginfo("  Row step: {}".format(msg.row_step))
    
    def publish_continuous(self, rate_hz=0.5):
        """
        Publish the point cloud continuously at specified rate
        
        Args:
            rate_hz (float): Publishing rate in Hz
        """
        rate = rospy.Rate(rate_hz)
        
        rospy.loginfo("Publishing continuously at {:.1f} Hz. Press Ctrl+C to stop".format(rate_hz))
        
        while not rospy.is_shutdown():
            msg = self.create_pointcloud2_msg()
            
            rospy.loginfo("Publishing point cloud with {:,} points...".format(len(self.points)))
            self.pub.publish(msg)
            
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        
        rospy.loginfo("Continuous publishing stopped")


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Publish PCD file as ROS PointCloud2')
    parser.add_argument('pcd_file', help='Path to PCD file')
    parser.add_argument('--topic', default='/input_pointcloud', 
                       help='ROS topic name to publish on (default: /input_pointcloud)')
    parser.add_argument('--rate', type=float, default=0.5,
                       help='Publishing rate in Hz (default: 0.5)')
    parser.add_argument('--frame-id', default='base_link',
                       help='Frame ID for point cloud (default: base_link)')
    parser.add_argument('--continuous', action='store_true',
                       help='Publish continuously (default: publish once)')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Initial delay before publishing in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    # Validate PCD file
    if not os.path.exists(args.pcd_file):
        rospy.logerr("PCD file not found: {}".format(args.pcd_file))
        return 1
    
    try:
        # Create publisher
        publisher = PCDPublisher(args.pcd_file, args.topic, args.frame_id)
        
        # Publish based on mode
        if args.continuous:
            publisher.publish_continuous(args.rate)
        else:
            publisher.publish_once(args.delay)
            
        return 0
        
    except rospy.ROSInterruptException:
        rospy.loginfo("PCD Publisher shutdown by user")
        return 0
    except Exception as e:
        rospy.logerr("Error: {}".format(e))
        import traceback
        rospy.logerr(traceback.format_exc())
        return 1


def main_default_file():
    """
    Main function that publishes the default Hole-1168_3d.pcd file
    This is a convenience function for the specific file mentioned in the request
    """
    # Default file path as specified in the request
    default_pcd_file = "/home/abdulla/codes/stata_mobile_robot_upgraded_ws/src/strata_robotic_machining/state_machine/scripts/csk_laser/november_11_cont_scan/Hole-1168_3d.pcd"
    
    try:
        # Create publisher with default settings
        publisher = PCDPublisher(default_pcd_file, '/input_pointcloud', 'base_link')
        
        # Publish once with a 2 second delay
        publisher.publish_once(delay=2.0)
        
        return 0
        
    except rospy.ROSInterruptException:
        rospy.loginfo("PCD Publisher shutdown by user")
        return 0
    except Exception as e:
        rospy.logerr("Error publishing default PCD file: {}".format(e))
        import traceback
        rospy.logerr(traceback.format_exc())
        return 1


if __name__ == '__main__':
    # Check if any arguments provided
    if len(sys.argv) == 1:
        # No arguments - use default file
        rospy.loginfo("No arguments provided, using default Hole-1168_3d.pcd file")
        sys.exit(main_default_file())
    else:
        # Arguments provided - use command line parsing
        sys.exit(main())
