#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import Image

from sensor_msgs_py import point_cloud2
import numpy as np
import tf2_py

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('Hybrid_Fusion')

        # Subscribing to the PointCloud2 messages coming from the Velodyne driver
        # They then get sent into the pointsCallback function
        self.pc2Subscriber = self.create_subscription(sensor_msgs.PointCloud2, "/velodyne_points", self.pointsCallback, 10)

        # Create subscription to the /raw_image
        self.imageSubscriber = self.create_subscription(Image, "raw_image", 10)

        # Create the publisher for the transformed matrix
        self.transformedPublisher = self.create_publisher(sensor_msgs.PointCloud2, '/transformed_point_cloud', 10)

        # Extrinsic transformation matrix
        self.extrinsic_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Camera intrinsic matrix in the shape:
        # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        self.camera_intrinsic = np.array([
            [1061.7, 0.0, 1095.6],
            [0.0, 1061.48, 613.6390],
            [0.0, 0.0, 1.0]
        ])
        # self.translation = np.array([1.0, 2.0, 3.0])  # Example translation
        # self.rotation_matrix = self.euler_to_rotation_matrix(np.pi / 4, np.pi / 4, np.pi / 4)  # Example rotation
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        # Convert Euler angles to rotation matrix
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx
    
    def create_cloud(self, header, points):
        # Create PointCloud2 message
        return point_cloud2.create_cloud_xyz32(header=header, points=points)

    def apply_transformation(self, points):

        # Convert the point cloud into a homogeneous coordinates representation for easy matrix math
        homoPoints = np.hstack([points, np.ones((points.shape[0], 1))])

        # Multiply the extrinsic tranformation matrix with the incoming lidar points
        # This will apply the Rotation and Translation
        transformed_point_cloud = self.extrinsic_matrix @ homoPoints.T

        # Convert back from homogeneous coordinates
        transformed_point_cloud = transformed_point_cloud[:3, :].T

        # Return the, now transformed, point cloud
        return transformed_point_cloud

    def pointsCallback(self, msg: sensor_msgs.PointCloud2):

        try:
            # Get the points from the lidar
            points = np.array(point_cloud2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True))

            if points.size == 0:
                self.get_logger().warn("Received empty point cloud")
                return
            
            # Apply the transformations to the points using the extrinsic matrix
            self.transformedPoints = self.apply_transformation(points)

            # Create a new PointCloud2 message from the transformed points
            transformed_cloud_msg = self.create_cloud(msg.header, self.transformedPoints)
            
            # Publish the transformed points
            self.transformedPublisher.publish(transformed_cloud_msg)
        
        except Exception as e:
            self.get_logger().error(f"Failed to process point cloud: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()