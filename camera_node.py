#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

rospy.init_node("camera_node")
bridge = CvBridge()

publisher = rospy.Publisher("/camera/image_raw", Image, queue_size=1)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 60)

rospy.loginfo("CamNode awake")

rate = rospy.Rate(60)

frame = 0

while not rospy.is_shutdown():
    ok, frame = cam.read()
    msg = bridge.cv2_to_imgmsg(frame, "bgr8")
    msg.header.stamp = rospy.get_rostime()
    msg.header.frame_id = "camera_node"

    publisher.publish(msg)

    frame += 1
    if frame % 60 == 0:
        rospy.loginfo("Frame: " + str(frame))

    rate.sleep()
