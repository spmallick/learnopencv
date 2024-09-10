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
 * $Id: extract_polygonal_prism_data.hpp 32996 2010-09-30 23:42:11Z rusu $
 *
 */

#include <pluginlib/class_list_macros.h>
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include "pcl_ros/transforms.hpp"
#include "pcl_ros/segmentation/extract_polygonal_prism_data.hpp"

using pcl_conversions::moveFromPCL;
using pcl_conversions::moveToPCL;

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ExtractPolygonalPrismData::onInit()
{
  // Call the super onInit ()
  PCLNodelet::onInit();

  // Enable the dynamic reconfigure service
  srv_ = boost::make_shared<dynamic_reconfigure::Server<ExtractPolygonalPrismDataConfig>>(*pnh_);
  dynamic_reconfigure::Server<ExtractPolygonalPrismDataConfig>::CallbackType f = boost::bind(
    &ExtractPolygonalPrismData::config_callback, this, _1, _2);
  srv_->setCallback(f);

  // Advertise the output topics
  pub_output_ = advertise<PointIndices>(*pnh_, "output", max_queue_size_);

  onInitPostProcess();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ExtractPolygonalPrismData::subscribe()
{
  sub_hull_filter_.subscribe(*pnh_, "planar_hull", max_queue_size_);
  sub_input_filter_.subscribe(*pnh_, "input", max_queue_size_);

  // Create the objects here
  if (approximate_sync_) {
    sync_input_hull_indices_a_ =
      boost::make_shared<message_filters::Synchronizer<
          sync_policies::ApproximateTime<PointCloud, PointCloud, PointIndices>>>(max_queue_size_);
  } else {
    sync_input_hull_indices_e_ =
      boost::make_shared<message_filters::Synchronizer<
          sync_policies::ExactTime<PointCloud, PointCloud, PointIndices>>>(max_queue_size_);
  }

  if (use_indices_) {
    sub_indices_filter_.subscribe(*pnh_, "indices", max_queue_size_);
    if (approximate_sync_) {
      sync_input_hull_indices_a_->connectInput(
        sub_input_filter_, sub_hull_filter_,
        sub_indices_filter_);
    } else {
      sync_input_hull_indices_e_->connectInput(
        sub_input_filter_, sub_hull_filter_,
        sub_indices_filter_);
    }
  } else {
    sub_input_filter_.registerCallback(bind(&ExtractPolygonalPrismData::input_callback, this, _1));

    if (approximate_sync_) {
      sync_input_hull_indices_a_->connectInput(sub_input_filter_, sub_hull_filter_, nf_);
    } else {
      sync_input_hull_indices_e_->connectInput(sub_input_filter_, sub_hull_filter_, nf_);
    }
  }
  // Register callbacks
  if (approximate_sync_) {
    sync_input_hull_indices_a_->registerCallback(
      bind(
        &ExtractPolygonalPrismData::
        input_hull_indices_callback, this, _1, _2, _3));
  } else {
    sync_input_hull_indices_e_->registerCallback(
      bind(
        &ExtractPolygonalPrismData::
        input_hull_indices_callback, this, _1, _2, _3));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ExtractPolygonalPrismData::unsubscribe()
{
  sub_hull_filter_.unsubscribe();
  sub_input_filter_.unsubscribe();

  if (use_indices_) {
    sub_indices_filter_.unsubscribe();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ExtractPolygonalPrismData::config_callback(
  ExtractPolygonalPrismDataConfig & config,
  uint32_t level)
{
  double height_min, height_max;
  impl_.getHeightLimits(height_min, height_max);
  if (height_min != config.height_min) {
    height_min = config.height_min;
    NODELET_DEBUG(
      "[%s::config_callback] Setting new minimum height to the planar model to: %f.",
      getName().c_str(), height_min);
    impl_.setHeightLimits(height_min, height_max);
  }
  if (height_max != config.height_max) {
    height_max = config.height_max;
    NODELET_DEBUG(
      "[%s::config_callback] Setting new maximum height to the planar model to: %f.",
      getName().c_str(), height_max);
    impl_.setHeightLimits(height_min, height_max);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ExtractPolygonalPrismData::input_hull_indices_callback(
  const PointCloudConstPtr & cloud,
  const PointCloudConstPtr & hull,
  const PointIndicesConstPtr & indices)
{
  // No subscribers, no work
  if (pub_output_.getNumSubscribers() <= 0) {
    return;
  }

  // Copy the header (stamp + frame_id)
  pcl_msgs::PointIndices inliers;
  inliers.header = fromPCL(cloud->header);

  // If cloud is given, check if it's valid
  if (!isValid(cloud) || !isValid(hull, "planar_hull")) {
    NODELET_ERROR("[%s::input_hull_indices_callback] Invalid input!", getName().c_str());
    pub_output_.publish(inliers);
    return;
  }
  // If indices are given, check if they are valid
  if (indices && !isValid(indices)) {
    NODELET_ERROR("[%s::input_hull_indices_callback] Invalid indices!", getName().c_str());
    pub_output_.publish(inliers);
    return;
  }

  /// DEBUG
  if (indices) {
    NODELET_DEBUG(
      "[%s::input_indices_hull_callback]\n"
      "                                 - PointCloud with %d data points (%s), stamp %f, and "
      "frame %s on topic %s received.\n"
      "                                 - PointCloud with %d data points (%s), stamp %f, and "
      "frame %s on topic %s received.\n"
      "                                 - PointIndices with %zu values, stamp %f, and "
      "frame %s on topic %s received.",
      getName().c_str(),
      cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
        "input").c_str(),
      hull->width * hull->height, pcl::getFieldsList(*hull).c_str(), fromPCL(
        hull->header).stamp.toSec(), hull->header.frame_id.c_str(), pnh_->resolveName(
        "planar_hull").c_str(),
      indices->indices.size(), indices->header.stamp.toSec(),
      indices->header.frame_id.c_str(), pnh_->resolveName("indices").c_str());
  } else {
    NODELET_DEBUG(
      "[%s::input_indices_hull_callback]\n"
      "                                 - PointCloud with %d data points (%s), stamp %f, and "
      "frame %s on topic %s received.\n"
      "                                 - PointCloud with %d data points (%s), stamp %f, and "
      "frame %s on topic %s received.",
      getName().c_str(),
      cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
        "input").c_str(),
      hull->width * hull->height, pcl::getFieldsList(*hull).c_str(), fromPCL(
        hull->header).stamp.toSec(), hull->header.frame_id.c_str(), pnh_->resolveName(
        "planar_hull").c_str());
  }
  ///

  if (cloud->header.frame_id != hull->header.frame_id) {
    NODELET_DEBUG(
      "[%s::input_hull_callback] Planar hull has a different TF frame (%s) than the input "
      "point cloud (%s)! Using TF to transform.",
      getName().c_str(), hull->header.frame_id.c_str(), cloud->header.frame_id.c_str());
    PointCloud planar_hull;
    if (!pcl_ros::transformPointCloud(cloud->header.frame_id, *hull, planar_hull, tf_listener_)) {
      // Publish empty before return
      pub_output_.publish(inliers);
      return;
    }
    impl_.setInputPlanarHull(pcl_ptr(planar_hull.makeShared()));
  } else {
    impl_.setInputPlanarHull(pcl_ptr(hull));
  }

  IndicesPtr indices_ptr;
  if (indices && !indices->header.frame_id.empty()) {
    indices_ptr.reset(new std::vector<int>(indices->indices));
  }

  impl_.setInputCloud(pcl_ptr(cloud));
  impl_.setIndices(indices_ptr);

  // Final check if the data is empty
  // (remember that indices are set to the size of the data -- if indices* = NULL)
  if (!cloud->points.empty()) {
    pcl::PointIndices pcl_inliers;
    moveToPCL(inliers, pcl_inliers);
    impl_.segment(pcl_inliers);
    moveFromPCL(pcl_inliers, inliers);
  }
  // Enforce that the TF frame and the timestamp are copied
  inliers.header = fromPCL(cloud->header);
  pub_output_.publish(inliers);
  NODELET_DEBUG(
    "[%s::input_hull_callback] Publishing %zu indices.",
    getName().c_str(), inliers.indices.size());
}

typedef pcl_ros::ExtractPolygonalPrismData ExtractPolygonalPrismData;
PLUGINLIB_EXPORT_CLASS(ExtractPolygonalPrismData, nodelet::Nodelet)
