// -*- mode: c++ -*-
/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2015, JSK Lab
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/o2r other materials provided
 *     with the distribution.
 *   * Neither the name of the JSK Lab nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/


#ifndef JSK_PERCEPTION_APPLY_MASK_IMAGE_H_
#define JSK_PERCEPTION_APPLY_MASK_IMAGE_H_

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <jsk_topic_tools/diagnostic_nodelet.h>
#include <sensor_msgs/Image.h>
#include <jsk_recognition_msgs/RectArray.h>
#include <jsk_recognition_msgs/ClassificationResult.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

namespace jsk_perception
{
  class MyTracker {
  private:
      static int next_id;
      int id;
      std::string obj_class;
      int n_miss_frame;
      cv::Rect2d rect;
      cv::Ptr<cv::Tracker> cv_tracker;

      int MAX_MISS_FRAME;
      double MIN_NEW_DETECT_INTERSECTION_RATE;
      cv::Size MAX_DETECT_SIZE;
  public:
      // フレーム画像と追跡対象(Rect)で初期化
      MyTracker(const cv::Mat& _frame, const cv::Rect2d& _rect, const std::string&, const int, const double, const cv::Size) ;
      // 次フレームを入力にして、追跡対象の追跡(true)
      // MAX_MISS_FRAME以上検出が登録されていない場合は追跡失敗(false)
      bool update(const cv::Mat& _frame);
      // 新しい検出(Rect)を登録。
      // 現在位置と近ければ受理してn_miss_frameをリセット(true)
      // そうでなければ(false)
      bool registerNewDetect(const cv::Rect2d& _new_detect, const std::string&);
      // trackerの現在地を_imageに書き込む
      void draw(cv::Mat& _image) const;

      static void convertJSKRectArrayToCvRect(
          const jsk_recognition_msgs::RectArray::ConstPtr&,
          std::vector<cv::Rect_<int> >&);
  };

  class DetectionStabilizer: public jsk_topic_tools::DiagnosticNodelet
  {
  public:
    typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image,
    jsk_recognition_msgs::RectArray,
    jsk_recognition_msgs::ClassificationResult > ApproximateSyncPolicy;
    typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::Image,
    jsk_recognition_msgs::RectArray,
    jsk_recognition_msgs::ClassificationResult >SyncPolicy;
    DetectionStabilizer(): DiagnosticNodelet("DetectionStabilizer") {}

    std::vector<MyTracker> trackers;
  protected:

    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();
    virtual void apply(
      const sensor_msgs::Image::ConstPtr& image_msg,
      const jsk_recognition_msgs::RectArray::ConstPtr& rects_msg,
      const jsk_recognition_msgs::ClassificationResult::ConstPtr& classificaiton);

    bool approximate_sync_;
    int queue_size_;
    double MIN_NEW_DETECT_INTERSECTION_RATE;
    int MAX_MISS_FRAME;
    int max_detect_size_x;
    int max_detect_size_y;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> > sync_;
    boost::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy> > async_;
    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    message_filters::Subscriber<jsk_recognition_msgs::RectArray> sub_rects_;
    message_filters::Subscriber<jsk_recognition_msgs::ClassificationResult> sub_classification_;
    ros::Publisher pub_image_;
    ros::Publisher pub_mask_;
    
  private:
    
  };

}

#endif
