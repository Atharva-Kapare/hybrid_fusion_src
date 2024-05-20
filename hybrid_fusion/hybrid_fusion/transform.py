#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import sensor_msgs.msg as sensor_msgs
from sensor_msgs_py import point_cloud2
import numpy as np
import tf2_py

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('Hybrid_Fusion')

        # Subscribing to the PointCloud2 messages coming from the Velodyne driver
        # They then get sent into the pointsCallback function
        self.pc2Subscriber = self.create_subscription(sensor_msgs.PointCloud2, "/velodyne_points", self.pointsCallback, 10)

        # Create the publisher for the transformed matrix
        self.transformedPublisher = self.create_publisher(sensor_msgs.PointCloud2, '/transformed_point_cloud', 10)

        # Define the transformation matrix here
        # Define your transformation matrix here
        self.translation = np.array([1.0, 2.0, 3.0])  # Example translation
        self.rotation_matrix = self.euler_to_rotation_matrix(np.pi / 4, np.pi / 4, np.pi / 4)  # Example rotation
    
    # def define_transformation_matrix(self):
    #     # Example transformation matrix (4x4)
    #     translation = np.array([1.0, 2.0, 3.0])  # Replace with your translation values
    #     rotation = np.array([
    #         [0.0, -1.0, 0.0],
    #         [1.0, 0.0, 0.0],
    #         [0.0, 0.0, 1.0]
    #     ])  # Replace with your rotation matrix

    #     # Construct a 4x4 transformation matrix
    #     transformation_matrix = np.eye(4)
    #     transformation_matrix[:3, :3] = rotation
    #     transformation_matrix[:3, 3] = translation
    #     return transformation_matrix

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
        # fields = [
        #     point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
        #     point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
        #     point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
        # ]
        # fields = [
        #     sensor_msgs.PointField('x', 0, point_cloud2.PointField.FLOAT32, 7),
        #     sensor_msgs.PointField('y', 4, point_cloud2.PointField.FLOAT32, 7),
        #     sensor_msgs.PointField('z', 8, point_cloud2.PointField.FLOAT32, 7),
        # ]
        # Create PointCloud2 message
        return point_cloud2.create_cloud_xyz32(header=header, points=points)

    def apply_transformation(self, points):

        homoPoints = np.hstack([points, np.ones((points.shape[0], 1))])

        # Test transformation matrix
        self.transformation_matrix = np.array([
            [1, 0, 0, 5],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        transformedMatrix = homoPoints.dot(self.transformation_matrix.T)

        return transformedMatrix[:, :3]

    def pointsCallback(self, msg: sensor_msgs.PointCloud2):

        # self.get_logger().info(str(msg))
        try:
            # cloud = point_cloud2.read_points_numpy(msg)
            # self.get_logger().info(msg._fields_and_field_types)
            points = np.array(point_cloud2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True))

            # points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            # points_array = np.array(points)

            # self.get_logger().info(f"Points array shape: {points_array.shape}")

            if points.size == 0:
                self.get_logger().warn("Received empty point cloud")
                return
            
            transformedPoints = self.apply_transformation(points)
            # print(transformedPoints[0])

            # # Ensure points_array is divisible by 3 for reshaping
            # num_points = points_array.size // 3
            # points_array = points_array[:num_points * 3].reshape(-1, 3)

            # self.get_logger().info(f"Reshaped points array shape: {points_array.shape}")

            # # Convert points to homogeneous coordinates (N x 4)
            # homogeneous_points = np.hstack((points_array, np.ones((num_points, 1))))

            # # Apply the transformation
            # transformed_points = homogeneous_points @ self.transformation_matrix.T

            # # Remove the homogeneous coordinate
            # transformed_points = transformed_points[:, :3]

            # # Create a new PointCloud2 message
            transformed_cloud_msg = self.create_cloud(msg.header, transformedPoints)
            
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