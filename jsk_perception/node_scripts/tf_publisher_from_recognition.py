#!/usr/bin/env python

import rospy
import message_filters
import tf
import numpy as np
import cv2
import cv_bridge

from image_geometry import PinholeCameraModel
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import RectArray, Rect, ClassificationResult
from sensor_msgs.msg import CameraInfo, Image
import pdb

class RecognisedTfPublisher(ConnectionBasedTransport):
    def __init__(self):
        super(RecognisedTfPublisher, self).__init__()
        self.pub_ = self.advertise('~output/camera_info', CameraInfo, queue_size=10)
        self.cam_model = PinholeCameraModel()
        self.br = tf.TransformBroadcaster()

    def subscribe(self):
        self.sub_cam_info = message_filters.Subscriber('~input/camera_info', CameraInfo)
        self.sub_rects = message_filters.Subscriber('~input/rect_array', RectArray)
        self.sub_depth = message_filters.Subscriber('~input/depth', Image)
        self.sub_recognition = message_filters.Subscriber('~input/recog', ClassificationResult)
        queue_size = rospy.get_param('~queue_size', 3)
        slop = rospy.get_param('~slop', 10)
        subs = [self.sub_cam_info, self.sub_rects, self.sub_depth, self.sub_recognition]

        sync = message_filters.ApproximateTimeSynchronizer(
                subs, queue_size, slop)
        sync.registerCallback(self._apply)


    def unsubscribe(self):
        self.sub_.unregister()
        self.sub_rects.unregister()

    def _apply(self, camera_info, rect_array, dep_imgmsg, recog):
        # getting depth image
        bridge = cv_bridge.CvBridge()
        dep_im = bridge.imgmsg_to_cv2(dep_imgmsg)
        # setting camera
        self.cam_model.fromCameraInfo(camera_info)
        # get center of array
        uv_list = []
        for rect in rect_array.rects:
            uv = ( rect.x + rect.width / 2 ,
                    rect.y + rect.height /2 )
            uv_list.append(uv)
        
        for each_uv, label in zip(uv_list, recog.label_names):
            # convert center point to 3d normal vector via camera_info   
            point3d = np.array(self.cam_model.projectPixelTo3dRay(each_uv))
            # get depth from depth image which is effector of 3d point 
            depth = dep_im[each_uv[::-1]]
            if not depth != depth :  # except NaN 
                point3d = depth * point3d
                # publish tf
                self.br.sendTransform(point3d,
                        tf.transformations.quaternion_from_euler(0,0,0),
                        camera_info.header.stamp,
                        label, #this tf
                        self.cam_model.tfFrame()) #parent tf 

                

if __name__ == '__main__':
    rospy.init_node('recognised_tf_publisher', log_level=rospy.DEBUG)
    recongnised_tf_publisher = RecognisedTfPublisher()
    rospy.spin()
