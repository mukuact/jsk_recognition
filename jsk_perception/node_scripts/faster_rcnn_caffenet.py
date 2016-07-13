#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import cv2
import numpy as np
import pdb

import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import rospy
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import Rect, RectArray, ClassificationResult
import message_filters


if 'FRCN_ROOT' not in os.environ:
    print("Please set 'FRCN_ROOT' environmental variable.")
    sys.exit(1)

FRCN_ROOT = os.environ['FRCN_ROOT']

sys.path.insert(0, os.path.join(FRCN_ROOT, 'caffe-fast-rcnn/python'))
sys.path.insert(0, os.path.join(FRCN_ROOT, 'lib'))

import caffe
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg



CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

    
def vis_detections(class_ind, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return [],([],[],[])
    out_rects = []
    indies = []
    names = []
    scores = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
	out_rects.append(
                Rect(x=bbox[0],y=bbox[1],
                    width=(bbox[2]-bbox[0]),height=(bbox[3]-bbox[1])))

        indies.append(class_ind)
        names.append(class_name)
        scores.append(score)
    return out_rects, (indies, names, scores)

class FastRCNN(ConnectionBasedTransport):

    def __init__(self):
        super(FastRCNN, self).__init__()
        self._setting_caffe()
        self._pub_array = self.advertise('~rect_array', RectArray, queue_size=1)
        self._pub_value = self.advertise('~output', ClassificationResult, queue_size=1)

    def subscribe(self):
        sub = message_filters.Subscriber('~input', Image)
        sub.registerCallback(self._detect)

    def unsubscribe(self):
        self._sub.unregister()

    def _detect(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        im = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')

        # detect object
        tic = time.clock()
        out_rects, out_values = self._detect_obj(im)
        toc = time.clock()
        rospy.logdebug("detection time is %f sec",toc - tic)

        # publish array
        ros_rect_array = RectArray()
        ros_rect_array.header = imgmsg.header 
        ros_rect_array.rects = out_rects
        self._pub_array.publish(ros_rect_array)
        # publish values
        cls_msg = ClassificationResult(
                header = imgmsg.header,
                labels = out_values[0],
                label_names = out_values[1],
                label_proba = out_values[2]
                )
        self._pub_value.publish(cls_msg)


    def _detect_obj(self, im):
        
        caffe.set_mode_gpu()
        scores, boxes = im_detect(self.net, im)

        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
	rects_list = []
        value_list = [list(),list(),list()]
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            rects,values = vis_detections(cls_ind, cls, dets, thresh=CONF_THRESH) 
            rects_list.extend(rects)
            for i, value in enumerate(value_list):
                value.extend(values[i])
        return rects_list,value_list


    def _setting_caffe(self):
        cfg.TEST.HAS_RPN = True

        prototxt = os.path.join(FRCN_ROOT, 'models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt')
        caffemodel = os.path.join(FRCN_ROOT,
            'data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel')
        self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        rospy.loginfo('Loaded network {:s}'.format(caffemodel))

        # Warmup on a dummy image
        im = 128 * np.ones((480,640, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _= im_detect(self.net, im)


def main():
    rospy.init_node('fast_rcnn_caffenet', log_level=rospy.DEBUG)
    fast_rcnn = FastRCNN()
    rospy.spin()

if __name__ == '__main__':
    
    if rospy.get_param('~cpu_mode',False):
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    main()
