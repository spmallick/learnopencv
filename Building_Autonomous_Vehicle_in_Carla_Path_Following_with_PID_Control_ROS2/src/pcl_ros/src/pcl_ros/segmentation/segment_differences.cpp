/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
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
 * $Id: segment_differences.cpp 35361 2011-01-20 04:34:49Z rusu $
 *
 */

#include <pluginlib/class_list_macros.h>
#include <pcl/common/io.h>
#include "pcl_ros/segmentation/segment_differences.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::SegmentDifferences::onInit()
{
  // Call the super onInit ()
  PCLNodelet::onInit();

  pub_output_ = advertise<PointCloud>(*pnh_, "output", max_queue_size_);

  NODELET_DEBUG(
    "[%s::onInit] Nodelet successfully created with the following parameters:\n"
    " - max_queue_size    : %d",
    getName().c_str(),
    max_queue_size_);

  onInitPostProcess();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::SegmentDifferences::subscribe()
{
  // Subscribe to the input using a filter
  sub_input_filter_.subscribe(*pnh_, "input", max_queue_size_);
  sub_target_filter_.subscribe(*pnh_, "target", max_queue_size_);

  // Enable the dynamic reconfigure service
  srv_ = boost::make_shared<dynamic_reconfigure::Server<SegmentDifferencesConfig>>(*pnh_);
  dynamic_reconfigure::Server<SegmentDifferencesConfig>::CallbackType f = boost::bind(
    &SegmentDifferences::config_callback, this, _1, _2);
  srv_->setCallback(f);

  if (approximate_sync_) {
    sync_input_target_a_ =
      boost::make_shared<message_filters::Synchronizer<
          sync_policies::ApproximateTime<PointCloud, PointCloud>>>(max_queue_size_);
    sync_input_target_a_->connectInput(sub_input_filter_, sub_target_filter_);
    sync_input_target_a_->registerCallback(
      bind(
        &SegmentDifferences::input_target_callback, this,
        _1, _2));
  } else {
    sync_input_target_e_ =
      boost::make_shared<message_filters::Synchronizer<
          sync_policies::ExactTime<PointCloud, PointCloud>>>(max_queue_size_);
    sync_input_target_e_->connectInput(sub_input_filter_, sub_target_filter_);
    sync_input_target_e_->registerCallback(
      bind(
        &SegmentDifferences::input_target_callback, this,
        _1, _2));
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::SegmentDifferences::unsubscribe()
{
  sub_input_filter_.unsubscribe();
  sub_target_filter_.unsubscribe();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::SegmentDifferences::config_callback(SegmentDifferencesConfig & config, uint32_t level)
{
  if (impl_.getDistanceThreshold() != config.distance_threshold) {
    impl_.setDistanceThreshold(config.distance_threshold);
    NODELET_DEBUG(
      "[%s::config_callback] Setting new distance threshold to: %f.",
      getName().c_str(), config.distance_threshold);
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::SegmentDifferences::input_target_callback(
  const PointCloudConstPtr & cloud,
  const PointCloudConstPtr & cloud_target)
{
  if (pub_output_.getNumSubscribers() <= 0) {
    return;
  }

  if (!isValid(cloud) || !isValid(cloud_target, "target")) {
    NODELET_ERROR("[%s::input_indices_callback] Invalid input!", getName().c_str());
    PointCloud output;
    output.header = cloud->header;
    pub_output_.publish(ros_ptr(output.makeShared()));
    return;
  }

  NODELET_DEBUG(
    "[%s::input_indices_callback]\n"
    "                                 - PointCloud with %d data points (%s), stamp %f, and "
    "frame %s on topic %s received.\n"
    "                                 - PointCloud with %d data points (%s), stamp %f, and "
    "frame %s on topic %s received.",
    getName().c_str(),
    cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
      cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
      "input").c_str(),
    cloud_target->width * cloud_target->height, pcl::getFieldsList(*cloud_target).c_str(),
    fromPCL(cloud_target->header).stamp.toSec(),
    cloud_target->header.frame_id.c_str(), pnh_->resolveName("target").c_str());

  impl_.setInputCloud(pcl_ptr(cloud));
  impl_.setTargetCloud(pcl_ptr(cloud_target));

  PointCloud output;
  impl_.segment(output);

  pub_output_.publish(ros_ptr(output.makeShared()));
  NODELET_DEBUG(
    "[%s::segmentAndPublish] Published PointCloud2 with %zu points and stamp %f on topic %s",
    getName().c_str(),
    output.points.size(), fromPCL(output.header).stamp.toSec(),
    pnh_->resolveName("output").c_str());
}

typedef pcl_ros::SegmentDifferences SegmentDifferences;
PLUGINLIB_EXPORT_CLASS(SegmentDifferences, nodelet::Nodelet)
