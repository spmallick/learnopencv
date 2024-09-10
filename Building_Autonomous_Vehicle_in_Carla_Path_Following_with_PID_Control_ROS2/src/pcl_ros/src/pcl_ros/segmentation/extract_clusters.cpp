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
 * $Id: extract_clusters.hpp 32052 2010-08-27 02:19:30Z rusu $
 *
 */

#include <pluginlib/class_list_macros.h>
#include <pcl/common/io.h>
#include <pcl/PointIndices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include "pcl_ros/segmentation/extract_clusters.hpp"


using pcl_conversions::fromPCL;
using pcl_conversions::moveFromPCL;
using pcl_conversions::toPCL;

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::EuclideanClusterExtraction::onInit()
{
  // Call the super onInit ()
  PCLNodelet::onInit();

  // ---[ Mandatory parameters
  double cluster_tolerance;
  if (!pnh_->getParam("cluster_tolerance", cluster_tolerance)) {
    NODELET_ERROR(
      "[%s::onInit] Need a 'cluster_tolerance' parameter to be set before continuing!",
      getName().c_str());
    return;
  }
  int spatial_locator;
  if (!pnh_->getParam("spatial_locator", spatial_locator)) {
    NODELET_ERROR(
      "[%s::onInit] Need a 'spatial_locator' parameter to be set before continuing!",
      getName().c_str());
    return;
  }

  // private_nh.getParam ("use_indices", use_indices_);
  pnh_->getParam("publish_indices", publish_indices_);

  if (publish_indices_) {
    pub_output_ = advertise<PointIndices>(*pnh_, "output", max_queue_size_);
  } else {
    pub_output_ = advertise<PointCloud>(*pnh_, "output", max_queue_size_);
  }

  // Enable the dynamic reconfigure service
  srv_ = boost::make_shared<dynamic_reconfigure::Server<EuclideanClusterExtractionConfig>>(*pnh_);
  dynamic_reconfigure::Server<EuclideanClusterExtractionConfig>::CallbackType f = boost::bind(
    &EuclideanClusterExtraction::config_callback, this, _1, _2);
  srv_->setCallback(f);

  NODELET_DEBUG(
    "[%s::onInit] Nodelet successfully created with the following parameters:\n"
    " - max_queue_size    : %d\n"
    " - use_indices       : %s\n"
    " - cluster_tolerance : %f\n",
    getName().c_str(),
    max_queue_size_,
    (use_indices_) ? "true" : "false", cluster_tolerance);

  // Set given parameters here
  impl_.setClusterTolerance(cluster_tolerance);

  onInitPostProcess();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::EuclideanClusterExtraction::subscribe()
{
  // If we're supposed to look for PointIndices (indices)
  if (use_indices_) {
    // Subscribe to the input using a filter
    sub_input_filter_.subscribe(*pnh_, "input", max_queue_size_);
    sub_indices_filter_.subscribe(*pnh_, "indices", max_queue_size_);

    if (approximate_sync_) {
      sync_input_indices_a_ =
        boost::make_shared<message_filters::Synchronizer<
            message_filters::sync_policies::ApproximateTime<
              PointCloud, PointIndices>>>(max_queue_size_);
      sync_input_indices_a_->connectInput(sub_input_filter_, sub_indices_filter_);
      sync_input_indices_a_->registerCallback(
        bind(
          &EuclideanClusterExtraction::
          input_indices_callback, this, _1, _2));
    } else {
      sync_input_indices_e_ =
        boost::make_shared<message_filters::Synchronizer<
            message_filters::sync_policies::ExactTime<PointCloud, PointIndices>>>(max_queue_size_);
      sync_input_indices_e_->connectInput(sub_input_filter_, sub_indices_filter_);
      sync_input_indices_e_->registerCallback(
        bind(
          &EuclideanClusterExtraction::
          input_indices_callback, this, _1, _2));
    }
  } else {
    // Subscribe in an old fashion to input only (no filters)
    sub_input_ =
      pnh_->subscribe<PointCloud>(
      "input", max_queue_size_,
      bind(&EuclideanClusterExtraction::input_indices_callback, this, _1, PointIndicesConstPtr()));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::EuclideanClusterExtraction::unsubscribe()
{
  if (use_indices_) {
    sub_input_filter_.unsubscribe();
    sub_indices_filter_.unsubscribe();
  } else {
    sub_input_.shutdown();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::EuclideanClusterExtraction::config_callback(
  EuclideanClusterExtractionConfig & config,
  uint32_t level)
{
  if (impl_.getClusterTolerance() != config.cluster_tolerance) {
    impl_.setClusterTolerance(config.cluster_tolerance);
    NODELET_DEBUG(
      "[%s::config_callback] Setting new clustering tolerance to: %f.",
      getName().c_str(), config.cluster_tolerance);
  }
  if (impl_.getMinClusterSize() != config.cluster_min_size) {
    impl_.setMinClusterSize(config.cluster_min_size);
    NODELET_DEBUG(
      "[%s::config_callback] Setting the minimum cluster size to: %d.",
      getName().c_str(), config.cluster_min_size);
  }
  if (impl_.getMaxClusterSize() != config.cluster_max_size) {
    impl_.setMaxClusterSize(config.cluster_max_size);
    NODELET_DEBUG(
      "[%s::config_callback] Setting the maximum cluster size to: %d.",
      getName().c_str(), config.cluster_max_size);
  }
  if (max_clusters_ != config.max_clusters) {
    max_clusters_ = config.max_clusters;
    NODELET_DEBUG(
      "[%s::config_callback] Setting the maximum number of clusters to extract to: %d.",
      getName().c_str(), config.max_clusters);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::EuclideanClusterExtraction::input_indices_callback(
  const PointCloudConstPtr & cloud, const PointIndicesConstPtr & indices)
{
  // No subscribers, no work
  if (pub_output_.getNumSubscribers() <= 0) {
    return;
  }

  // If cloud is given, check if it's valid
  if (!isValid(cloud)) {
    NODELET_ERROR("[%s::input_indices_callback] Invalid input!", getName().c_str());
    return;
  }
  // If indices are given, check if they are valid
  if (indices && !isValid(indices)) {
    NODELET_ERROR("[%s::input_indices_callback] Invalid indices!", getName().c_str());
    return;
  }

  /// DEBUG
  if (indices) {
    std_msgs::Header cloud_header = fromPCL(cloud->header);
    std_msgs::Header indices_header = indices->header;
    NODELET_DEBUG(
      "[%s::input_indices_callback]\n"
      "                                 - PointCloud with %d data points (%s), stamp %f, and "
      "frame %s on topic %s received.\n"
      "                                 - PointIndices with %zu values, stamp %f, and "
      "frame %s on topic %s received.",
      getName().c_str(),
      cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(),
      cloud_header.stamp.toSec(), cloud_header.frame_id.c_str(), pnh_->resolveName("input").c_str(),
      indices->indices.size(), indices_header.stamp.toSec(),
      indices_header.frame_id.c_str(), pnh_->resolveName("indices").c_str());
  } else {
    NODELET_DEBUG(
      "[%s::input_callback] PointCloud with %d data points, stamp %f, and frame %s on "
      "topic %s received.",
      getName().c_str(), cloud->width * cloud->height, fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
        "input").c_str());
  }
  ///

  IndicesPtr indices_ptr;
  if (indices) {
    indices_ptr.reset(new std::vector<int>(indices->indices));
  }

  impl_.setInputCloud(pcl_ptr(cloud));
  impl_.setIndices(indices_ptr);

  std::vector<pcl::PointIndices> clusters;
  impl_.extract(clusters);

  if (publish_indices_) {
    for (size_t i = 0; i < clusters.size(); ++i) {
      if (static_cast<int>(i) >= max_clusters_) {
        break;
      }
      // TODO(xxx): HACK!!! We need to change the PointCloud2 message to add for an incremental
      // sequence ID number.
      pcl_msgs::PointIndices ros_pi;
      moveFromPCL(clusters[i], ros_pi);
      ros_pi.header.stamp += ros::Duration(i * 0.001);
      pub_output_.publish(ros_pi);
    }

    NODELET_DEBUG(
      "[segmentAndPublish] Published %zu clusters (PointIndices) on topic %s",
      clusters.size(), pnh_->resolveName("output").c_str());
  } else {
    for (size_t i = 0; i < clusters.size(); ++i) {
      if (static_cast<int>(i) >= max_clusters_) {
        break;
      }
      PointCloud output;
      copyPointCloud(*cloud, clusters[i].indices, output);

      // PointCloud output_blob;     // Convert from the templated output to the PointCloud blob
      // pcl::toROSMsg (output, output_blob);
      // TODO(xxx): HACK!!! We need to change the PointCloud2 message to add for an incremental
      // sequence ID number.
      std_msgs::Header header = fromPCL(output.header);
      header.stamp += ros::Duration(i * 0.001);
      toPCL(header, output.header);
      // Publish a Boost shared ptr const data
      pub_output_.publish(ros_ptr(output.makeShared()));
      NODELET_DEBUG(
        "[segmentAndPublish] Published cluster %zu (with %zu values and stamp %f) on topic %s",
        i, clusters[i].indices.size(), header.stamp.toSec(), pnh_->resolveName("output").c_str());
    }
  }
}

typedef pcl_ros::EuclideanClusterExtraction EuclideanClusterExtraction;
PLUGINLIB_EXPORT_CLASS(EuclideanClusterExtraction, nodelet::Nodelet)
