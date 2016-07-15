#!/usr/bin/env python

import rospy
import message_filters
import tf
import cv2
import cv_bridge

from image_geometry import PinholeCameraModel
from jsk_topic_tools import ConnectionBasedTransport
from jsk_recognition_msgs.msg import RectArray, Rect, ClassificationResult
from sensor_msgs.msg import CameraInfo
import pdb

class RecognisedTfPublisher(ConnectionBasedTransport):
    def __init__(self):
        super(RecognisedTfPublisher, self).__init__()
        self.pub_ = self.advertise('~output/camera_info', CameraInfo, queue_size=10)
        self.cam_model = PinholeCameraModel()
        self.br = tf.TransformBroadcaster()

    def subscribe(self):
        self.sub_ = message_filters.Subscriber('~input/camera_info', CameraInfo)
        self.sub_rects = message_filters.Subscriber('~input/rect_array', RectArray)
        queue_size = rospy.get_param('~queue_size', 3)
        slop = rospy.get_param('~slop', 10)
        subs = [self.sub_,self.sub_rects]

        sync = message_filters.ApproximateTimeSynchronizer(
                subs, queue_size, slop)
        sync.registerCallback(self._apply)


    def unsubscribe(self):
        self.sub_.unregister()
        self.sub_rects.unregister()

    def _apply(self, camera_info, rect_array):
        # setting camera
        self.cam_model.fromCameraInfo(camera_info)
        # get center of array
        uv_list = []
        for rect in rect_array.rects:
            uv = ( rect.x + rect.width / 2 ,
                    rect.y + rect.height /2 )
            uv_list.append(uv)

        point3d = self.cam_model.projectPixelTo3dRay(uv_list[0])

        rospy.logdebug(point3d)
        self.br.sendTransform(point3d,
                tf.transformations.quaternion_from_euler(0,0,0),
                camera_info.header.stamp,
                "bottle", #this tf
                self.cam_model.tfFrame()) #parent tf 

                

if __name__ == '__main__':
    rospy.init_node('recognised_tf_publisher', log_level=rospy.DEBUG)
    recongnised_tf_publisher = RecognisedTfPublisher()
    rospy.spin()
