#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import eval
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
from cv_bridge import CvBridge
import numpy as np

class YolactObjectDetection():

    def __init__(self):
        self.cvImage=None
        self.colorImage =None
        self.imageOutput=rospy.Publisher('/camera/depth_color_data/image_raw',Image, queue_size=10)

    def depth_value_callback(self,msg_depth):     
        self.cvImage = CvBridge().imgmsg_to_cv2(msg_depth, msg_depth.encoding) 
        self.cvImage=np.array(self.cvImage,dtype=np.int16)    

    def object_detection_callback(self,msg_rgb): 
        self.colorImage = CvBridge().imgmsg_to_cv2(msg_rgb, msg_rgb.encoding)
        with torch.no_grad():
            self.colorImage=eval.evaluate(self.colorImage)
        self.imageOutput.publish(CvBridge().cv2_to_imgmsg(self.colorImage, msg_rgb.encoding))



if __name__ == "__main__":
    rospy.init_node('Instance_Segmentation')
    frame=YolactObjectDetection()
    imagedepthInput = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, frame.depth_value_callback)
    imagergbInput = rospy.Subscriber('/camera/color/image_raw', Image, frame.object_detection_callback)    
    rospy.spin()
