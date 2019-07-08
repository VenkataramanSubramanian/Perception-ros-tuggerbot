#!/usr/bin/env python
import eval
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class YolactObjectDetection():

    def __init__(self):
        self.cvImage=None
        self.colorImage =None
        self.imageOutput=rospy.Publisher('/camera/depth_color_data/image_raw',Image, queue_size=10)

    def depth_value_callback(self,msg_depth):      
        self.cvImage = CvBridge().imgmsg_to_cv2(msg_depth, msg_depth.encoding)
        

    def object_detection_callback(self,msg_rgb):
        with torch.no_grad():
            np_image=cv2.imread('/home/venkat/hockey_660_080212045524.jpg')
            self.colorImage=eval.evaluate(np_image)
        #self.imageOutput.publish(CvBridge().cv2_to_imgmsg(self.colorImage, msg_rgb.encoding))
        cv2.imshow('image',self.colorImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node('Instance_Segmentation')
    frame=YolactObjectDetection()
    #imagedepthInput = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, frame.depth_value_callback)
    imagergbInput = rospy.Subscriber('/camera/color/image_raw', Image, frame.object_detection_callback)    
    rospy.spin()
