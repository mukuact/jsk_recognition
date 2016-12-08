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

#include <iostream>
#include <string>
#include <ros/console.h>
#include "jsk_perception/detection_stabilizer.h"
#include <boost/assign.hpp>
#include <jsk_topic_tools/log_utils.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <cv_bridge/cv_bridge.h>
#include <jsk_recognition_utils/cv_utils.h>

namespace jsk_perception
{
  void DetectionStabilizer::onInit()
  {
    DiagnosticNodelet::onInit();
    pnh_->param("approximate_sync", approximate_sync_, false);
    pnh_->param("queue_size", queue_size_, 100);
    pnh_->param("max_detect_size_x", max_detect_size_x, 100);
    pnh_->param("max_detect_size_y", max_detect_size_y, 200);
    pnh_->param("max_miss_frame", MAX_MISS_FRAME, 10);
    pnh_->param("min_new_detect_intersection_rate", MIN_NEW_DETECT_INTERSECTION_RATE, 0.5);
    pub_image_ = advertise<sensor_msgs::Image>(
      *pnh_, "output/image", 1);
    pub_array_ = advertise<jsk_recognition_msgs::RectArray>(
      *pnh_, "output/rect_array", 1);
    pub_class_ = advertise<jsk_recognition_msgs::ClassificationResult>(
      *pnh_, "output", 1);

    onInitPostProcess();
  }

  void DetectionStabilizer::subscribe()
  {
    sub_image_.subscribe(*pnh_, "input", 1);
    sub_rects_.subscribe(*pnh_, "input/rect_array", 1);
    sub_classification_.subscribe(*pnh_, "input/classification", 1);
    if (approximate_sync_) {
      async_ = boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy> >(queue_size_);
      async_->connectInput(sub_image_, sub_rects_, sub_classification_);
      async_->registerCallback(boost::bind(&DetectionStabilizer::apply, this, _1, _2, _3));
    }
    else {
      sync_ = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(queue_size_);
      sync_->connectInput(sub_image_, sub_rects_, sub_classification_);
      sync_->registerCallback(boost::bind(&DetectionStabilizer::apply, this, _1, _2, _3));
    }
    ros::V_string names = boost::assign::list_of("~input")("~input/rect_array")("~input/classification");
    jsk_topic_tools::warnNoRemap(names);
  }

  void DetectionStabilizer::unsubscribe()
  {
    sub_image_.unsubscribe();
    sub_rects_.unsubscribe();
    sub_classification_.unsubscribe();
  }

  void DetectionStabilizer::apply(
    const sensor_msgs::Image::ConstPtr& image_msg,
    const jsk_recognition_msgs::RectArray::ConstPtr& rects_msg,
    const jsk_recognition_msgs::ClassificationResult::ConstPtr& classification)
  {
    vital_checker_->poke();

    std::vector<cv::Rect> cv_rects; 
    MyTracker::convertJSKRectArrayToCvRect(rects_msg, cv_rects);

    cv::Mat image;
    if (jsk_recognition_utils::isBGRA(image_msg->encoding)) {
      cv::Mat tmp_image = cv_bridge::toCvShare(image_msg, image_msg->encoding)->image;
      cv::cvtColor(tmp_image, image, cv::COLOR_BGRA2BGR);
    }
    else if (jsk_recognition_utils::isRGBA(image_msg->encoding)) {
      cv::Mat tmp_image = cv_bridge::toCvShare(image_msg, image_msg->encoding)->image;
      cv::cvtColor(tmp_image, image, cv::COLOR_RGBA2BGR);
    }
    else {  // BGR, RGB or GRAY
      image = cv_bridge::toCvShare(image_msg, image_msg->encoding)->image;
    }
    
    // check whether rects_msg and ClassificationResult have same result
    if (rects_msg->rects.size() != classification->label_names.size()){
      return;
    }
    
    // update tracking
    for (std::vector<MyTracker>::iterator t_it = trackers.begin(); t_it != trackers.end();){
        t_it = (t_it->update(image)) ? ++t_it : trackers.erase(t_it);
    }

    // make new tracker when new object is detected
    for (size_t k=0; k< classification->label_names.size(); k++){
      const cv::Rect& tmp_rect = cv_rects[k];
      const std::string& tmp_class = classification->label_names[k];

      bool all_exists = false;
      for (std::vector<MyTracker>::iterator t_it = trackers.begin(); t_it != trackers.end();t_it++){
        bool exists = t_it->registerNewDetect(tmp_rect, tmp_class);
        if (exists){
          all_exists = true; 
          break;
        }
      }
      if(!all_exists){
        MyTracker tmp(image, tmp_rect, tmp_class, MAX_MISS_FRAME, MIN_NEW_DETECT_INTERSECTION_RATE, cv::Size(max_detect_size_y, max_detect_size_x));
        trackers.push_back(tmp);
      }
    }

    // output
    cv::Mat output_image;
    output_image = image;

    std::vector<jsk_recognition_msgs::Rect> result_rects;
    std::vector<std::string> result_classes;

    for (std::vector<MyTracker>::iterator t_it = trackers.begin(); t_it != trackers.end();t_it++)
    {
        t_it->draw(output_image);
        t_it->output(result_rects, result_classes);
    }
    jsk_recognition_msgs::RectArray new_rects;
    new_rects.header = rects_msg->header;
    new_rects.rects = result_rects;

    jsk_recognition_msgs::ClassificationResult new_classificaiton;
    new_classificaiton.header = classification->header;
    new_classificaiton.label_names = result_classes;

    pub_image_.publish(cv_bridge::CvImage(
                image_msg->header,
                image_msg->encoding,
                output_image).toImageMsg());

    pub_array_.publish(new_rects);
    pub_class_.publish(new_classificaiton);
  }

  // class MyTracker
  MyTracker::MyTracker(const cv::Mat& _frame, const cv::Rect2d& _rect, const std::string& _class, const int _max_miss, const double _min_inter, const cv::Size _max_detect) 
      : id(next_id++), rect(_rect), obj_class(_class), n_miss_frame(0),
        MAX_MISS_FRAME(_max_miss), MIN_NEW_DETECT_INTERSECTION_RATE(_min_inter),
        MAX_DETECT_SIZE(_max_detect)
  {
    cv_tracker = cv::Tracker::create("KCF"); //  or "MIL"
    const cv::Mat tmp_mat = _frame.clone();
    cv_tracker->init(_frame, _rect);
  }

  bool MyTracker::update(const cv::Mat& _frame){
    n_miss_frame++;
    return cv_tracker->update(_frame, rect) && n_miss_frame < MAX_MISS_FRAME;
  }

  bool MyTracker::registerNewDetect(const cv::Rect2d& _new_detect, const std::string& _class){
    if (_class != obj_class){
      return false;
    }
    double intersection_rate = 1.0 * (_new_detect & rect).area() / (_new_detect | rect).area();
    bool is_registered = intersection_rate > MIN_NEW_DETECT_INTERSECTION_RATE;
    if (is_registered) n_miss_frame = 0;
    return is_registered;
  }

  void MyTracker::draw(cv::Mat& _image) const{
    cv::rectangle(_image, rect, cv::Scalar(255, 0, 0), 2, 1);
    cv::putText(_image, cv::format("%03d:%s", id,obj_class.c_str()), cv::Point(rect.x + 5, rect.y + 17), 
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1, CV_AA);
  }

  void MyTracker::output(std::vector<jsk_recognition_msgs::Rect>& out_rects, std::vector<std::string>& out_objclass)
  {
    jsk_recognition_msgs::Rect tmp_rect;   
    tmp_rect.x = rect.x;
    tmp_rect.y = rect.y;
    tmp_rect.width = rect.width;
    tmp_rect.height = rect.height;

    out_rects.push_back(tmp_rect);
    out_objclass.push_back(obj_class);
  }

  void MyTracker::convertJSKRectArrayToCvRect(
      const jsk_recognition_msgs::RectArray::ConstPtr& jsk_rects,
      std::vector<cv::Rect_<int> >& bounding_boxes)
  {
    for (std::vector<jsk_recognition_msgs::Rect>::const_iterator it = jsk_rects->rects.begin(); it != jsk_rects->rects.end(); it++) {
      cv::Rect2d tmp_rect;
      tmp_rect.x = it->x;
      tmp_rect.y = it->y;
      tmp_rect.width = it->width;
      tmp_rect.height = it->height;
      bounding_boxes.push_back(tmp_rect);
    }
  }
  
  int MyTracker::next_id;
}


#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS (jsk_perception::DetectionStabilizer, nodelet::Nodelet);
