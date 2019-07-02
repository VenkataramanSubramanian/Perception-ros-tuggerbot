#!/usr/bin/env python

import darknet
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
import cv2

class YoloObjectDetection():

    def __init__(self):
        self.cvImage=None
        self.colorImage=None
        self.imageOutput=rospy.Publisher('/camera/depth_color_data/image_raw',Image, queue_size=1)
       

    def depth_value_callback(self,msg_depth):      
        self.cvImage = CvBridge().imgmsg_to_cv2(msg_depth, msg_depth.encoding)
        

    def object_detection_callback(self,msg_rgb):

        self.colorImage = CvBridge().imgmsg_to_cv2(msg_rgb, msg_rgb.encoding)
        res=darknet.detect(self.colorImage)

        assert self.cvImage.shape==self.colorImage.shape[0:2]

        for i in res:
            detection=i[0].decode('utf-8')
            if(detection in ('person')):
                centre_x = int(i[2][0])
                centre_y = int(i[2][1])
                depth = self.cvImage[centre_x][centre_y]
                upper = (int(i[2][0]-i[2][2]/2),int(i[2][1]-i[2][3]/2))
                lower = (int(i[2][0]+i[2][2]/2),int(i[2][1]+i[2][3]/2))
                cv2.rectangle(self.colorImage, upper, lower , (255,0,0), thickness = 4)
                cv2.rectangle(self.colorImage, (int(i[2][0]-i[2][2]/2),int(i[2][1]-i[2][3]/2-20)), 
                                        (int(i[2][0]-i[2][2]/2 + 120),int(i[2][1]-i[2][3]/2)), (255,0,0), thickness = -1)
                cv2.putText(self.colorImage, '{0} {1:.2f} {2:.2f} in cm'.format(i[0].decode('utf-8'), i[1], depth/10),(int(i[2][0]-i[2][2]/2), 
                       int(i[2][1]-i[2][3]/2)  -6),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 0),1,cv2.LINE_AA)              
                cv2.circle(self.colorImage, (int(i[2][0]),int(i[2][1])), 2, (255,0,0), thickness=4)
        
        self.imageOutput.publish(CvBridge().cv2_to_imgmsg(self.colorImage, msg_rgb.encoding ))
        

if __name__ == "__main__":
    rospy.init_node('objdet_n' )
    frame=YoloObjectDetection()
    imagedepthInput = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, frame.depth_value_callback)
    imagergbInput = rospy.Subscriber('/camera/color/image_raw', Image, frame.object_detection_callback)
    
    rospy.spin()

