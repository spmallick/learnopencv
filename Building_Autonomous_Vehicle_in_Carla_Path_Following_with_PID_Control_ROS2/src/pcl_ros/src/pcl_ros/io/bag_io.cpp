/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
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
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
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
 *
 * $Id: bag_io.cpp 34896 2010-12-19 06:21:42Z rusu $
 *
 */

#include <pluginlib/class_list_macros.h>
#include <string>
#include "pcl_ros/io/bag_io.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl_ros::BAGReader::open(const std::string & file_name, const std::string & topic_name)
{
  try {
    bag_.open(file_name, rosbag::bagmode::Read);
    view_.addQuery(bag_, rosbag::TopicQuery(topic_name));

    if (view_.size() == 0) {
      return false;
    }

    it_ = view_.begin();
  } catch (rosbag::BagException & e) {
    return false;
  }
  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::BAGReader::onInit()
{
  boost::shared_ptr<ros::NodeHandle> pnh_;
  pnh_.reset(new ros::NodeHandle(getMTPrivateNodeHandle()));
  // ---[ Mandatory parameters
  if (!pnh_->getParam("file_name", file_name_)) {
    NODELET_ERROR("[onInit] Need a 'file_name' parameter to be set before continuing!");
    return;
  }
  if (!pnh_->getParam("topic_name", topic_name_)) {
    NODELET_ERROR("[onInit] Need a 'topic_name' parameter to be set before continuing!");
    return;
  }
  // ---[ Optional parameters
  int max_queue_size = 1;
  pnh_->getParam("publish_rate", publish_rate_);
  pnh_->getParam("max_queue_size", max_queue_size);

  ros::Publisher pub_output = pnh_->advertise<sensor_msgs::PointCloud2>("output", max_queue_size);

  NODELET_DEBUG(
    "[onInit] Nodelet successfully created with the following parameters:\n"
    " - file_name    : %s\n"
    " - topic_name   : %s",
    file_name_.c_str(), topic_name_.c_str());

  if (!open(file_name_, topic_name_)) {
    return;
  }
  PointCloud output;
  output_ = boost::make_shared<PointCloud>(output);
  output_->header.stamp = ros::Time::now();

  // Continous publishing enabled?
  while (pnh_->ok()) {
    PointCloudConstPtr cloud = getNextCloud();
    NODELET_DEBUG(
      "Publishing data (%d points) on topic %s in frame %s.",
      output_->width * output_->height, pnh_->resolveName(
        "output").c_str(), output_->header.frame_id.c_str());
    output_->header.stamp = ros::Time::now();

    pub_output.publish(output_);

    ros::Duration(publish_rate_).sleep();
    ros::spinOnce();
  }
}

typedef pcl_ros::BAGReader BAGReader;
PLUGINLIB_EXPORT_CLASS(BAGReader, nodelet::Nodelet);
