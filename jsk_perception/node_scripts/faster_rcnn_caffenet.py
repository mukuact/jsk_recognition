#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import cv2
import numpy as np

import cv_bridge
from jsk_topic_tools import ConnectionBasedTransport
import rospy
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import RectArray


if 'FRCN_ROOT' not in os.environ:
    print("Please set 'FRCN_ROOT' environmental variable.")
    sys.exit(1)

FRCN_ROOT = os.environ['FRCN_ROOT']
sys.path.insert(0, os.path.join(FRCN_ROOT, 'caffe-fast-rcnn/python'))
sys.path.insert(0, os.path.join(FRCN_ROOT, 'lib'))

print(sys.path)
import caffe
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg


caffe.set_mode_gpu()
caffe.set_device(0)

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im

    out = im.copy()

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
        text = '{:s} {:.3f}'.format(class_name, score)
        cv2.putText(out, text,
                    (int(bbox[0]), int(bbox[1]) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    return out

class FastRCNN(ConnectionBasedTransport):

    def __init__(self, net):
        super(FastRCNN, self).__init__()
        self.net = net
        self._pub = self.advertise('~output', Image, queue_size=1)

    def subscribe(self):
        self._sub = rospy.Subscriber('~input', Image, self._detect)

    def unsubscribe(self):
        self._sub.unregister()
        self._sub_rects.unregister()

    def _detect(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        im = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')
        out_im = self._detect_obj(im)
        out_msg = bridge.cv2_to_imgmsg(out_im, encoding='bgr8')
        out_msg.header = imgmsg.header
        self._pub.publish(out_msg)

    def _detect_obj(self, im):
        scores, boxes = im_detect(self.net, im)

        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            im = vis_detections(im, cls, dets, thresh=CONF_THRESH)
        return im


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True
    prototxt = os.path.join(FRCN_ROOT, 'models/pascal_voc/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt')
    caffemodel = os.path.join(FRCN_ROOT,
        'data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel')
    caffenet = caffe.Net(prototxt, caffemodel, caffe.TEST)

    rospy.init_node('fast_rcnn_caffenet')
    fast_rcnn = FastRCNN(net=caffenet)
    rospy.spin()
