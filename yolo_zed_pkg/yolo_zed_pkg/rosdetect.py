#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import torch
import numpy as np
from pyzed import sl
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from my_robot_interfaces.msg import Cone, Cones

class YoloZedNode(Node):
    def __init__(self):
        super().__init__('yolo_zed_node')
        
        # Create a publisher for detected objects information
        self.pub = self.create_publisher(Cones, 'detected_objects', 10)
        self.pubString = self.create_publisher(String, 'detConesStr', 10)
        self.raw_image_pub = self.create_publisher(Image, 'raw_image', 10)
        
        # Initialize ZED camera
        init = sl.InitParameters()
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init.coordinate_units = sl.UNIT.CENTIMETER
        
        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("Error opening ZED camera. Exiting...")
            rclpy.shutdown()
            return
        
        self.runtime = sl.RuntimeParameters()
        self.mat_l = sl.Mat()
        self.mat_r = sl.Mat()
        self.depth_map = sl.Mat()
        self.point_cloud = sl.Mat()
        
        # Load YOLOv5 model
        self.model = torch.hub.load('/home/utsma/Desktop/Velo_Workspace/src/yolo_zed_pkg/yolo_zed_pkg/yolov5', 'custom', '/home/utsma/Desktop/Velo_Workspace/src/yolo_zed_pkg/yolo_zed_pkg/yolov5/best.pt', source='local', force_reload=True)
        self.bridge = CvBridge()
        
        # Create a timer to call the capture_and_detect method periodically
        self.timer = self.create_timer(0.1, self.capture_and_detect)  # 10 Hz

    def capture_and_detect(self):
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            # Obtain left and right images
            self.zed.retrieve_image(self.mat_l, sl.VIEW.LEFT)

            # We're only using the left image for object detection
            # self.zed.retrieve_image(self.mat_r, sl.VIEW.RIGHT)
            
            # Input the left image into the YOLOv5 model for object detection
            img_l = cv2.cvtColor(self.mat_l.get_data(), cv2.COLOR_RGBA2RGB)
            img_1 = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
            results = self.model(img_1, size=640)
            
            # Obtain depth map
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            
            targets_info = []
            
            # Traverse detected targets and draw bounding boxes
            for *xyxy, conf, cls in results.xyxy[0]:
                
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)
                
                coord_val = self.depth_map.get_value(x_center, y_center)
                x, y, z = self.point_cloud.get_value(x_center, y_center)[1][:3]
                
                # Virgil's original code
                # targets_info.append(f"{x}, {y}, {z}, {int(cls)}")


                cone = Cone()
                cone.x1 = int(xyxy[0])
                cone.y1 = int(xyxy[1])
                cone.x2 = int(xyxy[2])
                cone.y2 = int(xyxy[3])

                targets_info.append(cone)

                cv2.rectangle(img_l, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(img_l, results.names[int(cls)], (int(xyxy[0]), int(xyxy[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Publish detected objects information
            cones = Cones()
            cones.cones = targets_info
            # self.pub.publish(String(data=str(targets_info)))
            self.pub.publish(cones)
            self.pubString.publish(String(data=str(cones)))
            image_msg = self.bridge.cv2_to_imgmsg(img_l)
            self.raw_image_pub.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloZedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
