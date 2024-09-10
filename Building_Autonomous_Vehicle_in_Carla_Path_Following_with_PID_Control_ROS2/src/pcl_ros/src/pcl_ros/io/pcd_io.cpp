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
 * $Id: pcd_io.cpp 35812 2011-02-08 00:05:03Z rusu $
 *
 */

#include <pluginlib/class_list_macros.h>
#include <string>
#include <pcl_ros/io/pcd_io.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PCDReader::onInit()
{
  PCLNodelet::onInit();
  // Provide a latched topic
  ros::Publisher pub_output = pnh_->advertise<PointCloud2>("output", max_queue_size_, true);

  pnh_->getParam("publish_rate", publish_rate_);
  pnh_->getParam("tf_frame", tf_frame_);

  NODELET_DEBUG(
    "[%s::onInit] Nodelet successfully created with the following parameters:\n"
    " - publish_rate : %f\n"
    " - tf_frame     : %s",
    getName().c_str(),
    publish_rate_, tf_frame_.c_str());

  PointCloud2Ptr output_new;
  output_ = boost::make_shared<PointCloud2>();
  output_new = boost::make_shared<PointCloud2>();

  // Wait in a loop until someone connects
  do {
    ROS_DEBUG_ONCE("[%s::onInit] Waiting for a client to connect...", getName().c_str());
    ros::spinOnce();
    ros::Duration(0.01).sleep();
  } while (pnh_->ok() && pub_output.getNumSubscribers() == 0);

  std::string file_name;

  while (pnh_->ok()) {
    // Get the current filename parameter. If no filename set, loop
    if (!pnh_->getParam("filename", file_name_) && file_name_.empty()) {
      ROS_ERROR_ONCE(
        "[%s::onInit] Need a 'filename' parameter to be set before continuing!",
        getName().c_str());
      ros::Duration(0.01).sleep();
      ros::spinOnce();
      continue;
    }

    // If the filename parameter holds a different value than the last one we read
    if (file_name_.compare(file_name) != 0 && !file_name_.empty()) {
      NODELET_INFO("[%s::onInit] New file given: %s", getName().c_str(), file_name_.c_str());
      file_name = file_name_;
      pcl::PCLPointCloud2 cloud;
      if (impl_.read(file_name_, cloud) < 0) {
        NODELET_ERROR("[%s::onInit] Error reading %s !", getName().c_str(), file_name_.c_str());
        return;
      }
      pcl_conversions::moveFromPCL(cloud, *(output_));
      output_->header.stamp = ros::Time::now();
      output_->header.frame_id = tf_frame_;
    }

    // We do not publish empty data
    if (output_->data.size() == 0) {
      continue;
    }

    if (publish_rate_ == 0) {
      if (output_ != output_new) {
        NODELET_DEBUG(
          "Publishing data once (%d points) on topic %s in frame %s.",
          output_->width * output_->height,
          getMTPrivateNodeHandle().resolveName("output").c_str(), output_->header.frame_id.c_str());
        pub_output.publish(output_);
        output_new = output_;
      }
      ros::Duration(0.01).sleep();
    } else {
      NODELET_DEBUG(
        "Publishing data (%d points) on topic %s in frame %s.",
        output_->width * output_->height, getMTPrivateNodeHandle().resolveName(
          "output").c_str(), output_->header.frame_id.c_str());
      output_->header.stamp = ros::Time::now();
      pub_output.publish(output_);

      ros::Duration(publish_rate_).sleep();
    }

    ros::spinOnce();
    // Update parameters from server
    pnh_->getParam("publish_rate", publish_rate_);
    pnh_->getParam("tf_frame", tf_frame_);
  }

  onInitPostProcess();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PCDWriter::onInit()
{
  PCLNodelet::onInit();

  sub_input_ = pnh_->subscribe("input", 1, &PCDWriter::input_callback, this);
  // ---[ Optional parameters
  pnh_->getParam("filename", file_name_);
  pnh_->getParam("binary_mode", binary_mode_);

  NODELET_DEBUG(
    "[%s::onInit] Nodelet successfully created with the following parameters:\n"
    " - filename     : %s\n"
    " - binary_mode  : %s",
    getName().c_str(),
    file_name_.c_str(), (binary_mode_) ? "true" : "false");

  onInitPostProcess();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PCDWriter::input_callback(const PointCloud2ConstPtr & cloud)
{
  if (!isValid(cloud)) {
    return;
  }

  pnh_->getParam("filename", file_name_);

  NODELET_DEBUG(
    "[%s::input_callback] PointCloud with %d data points and frame %s on topic %s received.",
    getName().c_str(), cloud->width * cloud->height,
    cloud->header.frame_id.c_str(), getMTPrivateNodeHandle().resolveName("input").c_str());

  std::string fname;
  if (file_name_.empty()) {
    fname = boost::lexical_cast<std::string>(cloud->header.stamp.toSec()) + ".pcd";
  } else {
    fname = file_name_;
  }
  pcl::PCLPointCloud2 pcl_cloud;
  // It is safe to remove the const here because we are the only subscriber callback.
  pcl_conversions::moveToPCL(*(const_cast<PointCloud2 *>(cloud.get())), pcl_cloud);
  impl_.write(
    fname, pcl_cloud, Eigen::Vector4f::Zero(),
    Eigen::Quaternionf::Identity(), binary_mode_);

  NODELET_DEBUG("[%s::input_callback] Data saved to %s", getName().c_str(), fname.c_str());
}

typedef pcl_ros::PCDReader PCDReader;
typedef pcl_ros::PCDWriter PCDWriter;
PLUGINLIB_EXPORT_CLASS(PCDReader, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(PCDWriter, nodelet::Nodelet);
