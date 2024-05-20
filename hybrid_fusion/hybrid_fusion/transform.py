#!/usr/bin/env python3

import math
import os
import rclpy
from rclpy.node import Node

import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import Image

from sensor_msgs_py import point_cloud2
import numpy as np
import tf2_py

from cv_bridge import CvBridge
import cv2

class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__('Hybrid_Fusion')

        # Subscribing to the PointCloud2 messages coming from the Velodyne driver
        # They then get sent into the pointsCallback function
        self.pc2Subscriber = self.create_subscription(sensor_msgs.PointCloud2, "/velodyne_points", self.pointsCallback, 10)

        # Create subscription to the /raw_image
        self.imageSubscriber = self.create_subscription(Image, "raw_image", self.imageCallback, 10)

        # Create an image publisher
        self.imagePublisher = self.create_publisher(Image, "/fusedImage", 10)

        # Create the publisher for the transformed matrix
        self.transformedPublisher = self.create_publisher(sensor_msgs.PointCloud2, '/transformed_point_cloud', 10)

        # Camera intrinsic matrix in the shape:
        # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        self.camera_intrinsic = np.array([
            [-531.330, 0.0, 632.628],
            [0.0, 531.3, 347.529],
            [0.0, 0.0, 1.0]
        ])

        # Translation of LIDAR relative to the camera
        self.translation = np.array([0, -0.25, 0.0])
        # Rotation matrix of LIDAR relative to camera
        self.roll = -90
        self.pitch = 90
        self.yaw = 0
        self.rotation_matrix = self.euler_to_rotation_matrix(math.radians(self.roll), math.radians(self.pitch), math.radians(self.yaw))

        # Extrinsic transformation matrix
        self.transformationMatrix = self.createTransformationMatrix()

        self.bridge = CvBridge()

        self.transformedPoints = None

    def createTransformationMatrix(self):
        rotation_matrix = self.rotation_matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = self.translation
        return transformation_matrix
    
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
        # return Rx @ Ry @ Rz
    
    def create_cloud(self, header, points):
        # Create PointCloud2 message
        return point_cloud2.create_cloud_xyz32(header=header, points=points)

    def apply_transformation(self, points):

        # Convert the point cloud into a homogeneous coordinates representation for easy matrix math
        homoPoints = np.hstack([points, np.ones((points.shape[0], 1))])

        # Multiply the extrinsic tranformation matrix with the incoming lidar points
        # This will apply the Rotation and Translation
        transformed_point_cloud = (self.transformationMatrix @ homoPoints.T)


        # Convert back from homogeneous coordinates
        transformed_point_cloud = transformed_point_cloud[:3, :].T

        # Return the, now transformed, point cloud
        return transformed_point_cloud

    def imageCallback(self, msg: Image):
        # This message will apply the current transformed matrix to the latest image and publish it
        image = self.bridge.imgmsg_to_cv2(msg)

        # image = np.zeros((1920,1080,3), np.uint8)

        if self.transformedPoints is not None:
            # Calculate the projected points with the intrinsic matrix
            points_projected = (self.camera_intrinsic @ self.transformedPoints.T)

            # Normalize the points to convert from homogeneous coordinates to 2D
            points_projected = points_projected / points_projected[2]

            # Convert from homogeneous coordinates to 2D
            points_projected = points_projected[:2, :]

            image_width = 1280
            image_height = 720
            points_projected = np.clip(points_projected.T, [0, 0], [image_width, image_height])

            # print("Projected Points: ", points_projected[0])
            # points_projected = points_projected[:, :2] / points_projected[:, 2, np.newaxis]

            # print(points_projected[0])

            # Visualize on the camera image
            for point in points_projected:
                x, y = int(point[0]), int(point[1])
                # print("X:", x, "Y:", y)
                # print("Image Shape:", image.shape[1], image.shape[0])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            
            cv2.circle(image, (1280,720), 10, (0,0,255), -1)

            self.imagePublisher.publish(self.bridge.cv2_to_imgmsg(image, encoding="bgr8"))

        return 

    def pointsCallback(self, msg: sensor_msgs.PointCloud2):

        try:
            # Get the points from the lidar
            points = np.array(point_cloud2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True))

            if points.size == 0:
                self.get_logger().warn("Received empty point cloud")
                return
            


            # file_path = os.path.abspath('test.npy')
            # # Save the array to a binary file
            # try:
            #     np.save(file_path, self.transformedPoints)
            #     self.get_logger().info(f"Array saved successfully at {file_path}")
            # except Exception as e:
            #     self.get_logger().error(f"An error occurred while saving the array: {e}")

            
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