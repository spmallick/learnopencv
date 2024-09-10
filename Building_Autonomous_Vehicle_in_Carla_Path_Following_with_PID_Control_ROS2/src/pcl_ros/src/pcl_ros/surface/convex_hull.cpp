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
 * $Id: convex_hull.hpp 32993 2010-09-30 23:08:57Z rusu $
 *
 */

#include <pluginlib/class_list_macros.h>
#include <pcl/common/io.h>
#include <geometry_msgs/PolygonStamped.h>
#include <vector>
#include "pcl_ros/surface/convex_hull.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ConvexHull2D::onInit()
{
  PCLNodelet::onInit();

  pub_output_ = advertise<PointCloud>(*pnh_, "output", max_queue_size_);
  pub_plane_ = advertise<geometry_msgs::PolygonStamped>(*pnh_, "output_polygon", max_queue_size_);

  // ---[ Optional parameters
  pnh_->getParam("use_indices", use_indices_);

  NODELET_DEBUG(
    "[%s::onInit] Nodelet successfully created with the following parameters:\n"
    " - use_indices    : %s",
    getName().c_str(),
    (use_indices_) ? "true" : "false");

  onInitPostProcess();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ConvexHull2D::subscribe()
{
  // If we're supposed to look for PointIndices (indices)
  if (use_indices_) {
    // Subscribe to the input using a filter
    sub_input_filter_.subscribe(*pnh_, "input", 1);
    // If indices are enabled, subscribe to the indices
    sub_indices_filter_.subscribe(*pnh_, "indices", 1);

    if (approximate_sync_) {
      sync_input_indices_a_ =
        boost::make_shared<message_filters::Synchronizer<
            sync_policies::ApproximateTime<PointCloud, PointIndices>>>(max_queue_size_);
      // surface not enabled, connect the input-indices duo and register
      sync_input_indices_a_->connectInput(sub_input_filter_, sub_indices_filter_);
      sync_input_indices_a_->registerCallback(
        bind(
          &ConvexHull2D::input_indices_callback, this, _1,
          _2));
    } else {
      sync_input_indices_e_ =
        boost::make_shared<message_filters::Synchronizer<
            sync_policies::ExactTime<PointCloud, PointIndices>>>(max_queue_size_);
      // surface not enabled, connect the input-indices duo and register
      sync_input_indices_e_->connectInput(sub_input_filter_, sub_indices_filter_);
      sync_input_indices_e_->registerCallback(
        bind(
          &ConvexHull2D::input_indices_callback, this, _1,
          _2));
    }
  } else {
    // Subscribe in an old fashion to input only (no filters)
    sub_input_ =
      pnh_->subscribe<PointCloud>(
      "input", 1,
      bind(&ConvexHull2D::input_indices_callback, this, _1, PointIndicesConstPtr()));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ConvexHull2D::unsubscribe()
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
pcl_ros::ConvexHull2D::input_indices_callback(
  const PointCloudConstPtr & cloud,
  const PointIndicesConstPtr & indices)
{
  // No subscribers, no work
  if (pub_output_.getNumSubscribers() <= 0 && pub_plane_.getNumSubscribers() <= 0) {
    return;
  }

  PointCloud output;

  // If cloud is given, check if it's valid
  if (!isValid(cloud)) {
    NODELET_ERROR("[%s::input_indices_callback] Invalid input!", getName().c_str());
    // Publish an empty message
    output.header = cloud->header;
    pub_output_.publish(ros_ptr(output.makeShared()));
    return;
  }
  // If indices are given, check if they are valid
  if (indices && !isValid(indices, "indices")) {
    NODELET_ERROR("[%s::input_indices_callback] Invalid indices!", getName().c_str());
    // Publish an empty message
    output.header = cloud->header;
    pub_output_.publish(ros_ptr(output.makeShared()));
    return;
  }

  /// DEBUG
  if (indices) {
    NODELET_DEBUG(
      "[%s::input_indices_model_callback]\n"
      "                                 - PointCloud with %d data points (%s), stamp %f, and "
      "frame %s on topic %s received.\n"
      "                                 - PointIndices with %zu values, stamp %f, and "
      "frame %s on topic %s received.",
      getName().c_str(),
      cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(),
      getMTPrivateNodeHandle().resolveName("input").c_str(),
      indices->indices.size(), indices->header.stamp.toSec(),
      indices->header.frame_id.c_str(), getMTPrivateNodeHandle().resolveName("indices").c_str());
  } else {
    NODELET_DEBUG(
      "[%s::input_indices_callback] PointCloud with %d data points, stamp %f, and "
      "frame %s on topic %s received.",
      getName().c_str(), cloud->width * cloud->height, fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(),
      getMTPrivateNodeHandle().resolveName("input").c_str());
  }

  // Reset the indices and surface pointers
  IndicesPtr indices_ptr;
  if (indices) {
    indices_ptr.reset(new std::vector<int>(indices->indices));
  }

  impl_.setInputCloud(pcl_ptr(cloud));
  impl_.setIndices(indices_ptr);

  // Estimate the feature
  impl_.reconstruct(output);

  // If more than 3 points are present, send a PolygonStamped hull too
  if (output.points.size() >= 3) {
    geometry_msgs::PolygonStamped poly;
    poly.header = fromPCL(output.header);
    poly.polygon.points.resize(output.points.size());
    // Get three consecutive points (without copying)
    pcl::Vector4fMap O = output.points[1].getVector4fMap();
    pcl::Vector4fMap B = output.points[0].getVector4fMap();
    pcl::Vector4fMap A = output.points[2].getVector4fMap();
    // Check the direction of points -- polygon must have CCW direction
    Eigen::Vector4f OA = A - O;
    Eigen::Vector4f OB = B - O;
    Eigen::Vector4f N = OA.cross3(OB);
    double theta = N.dot(O);
    bool reversed = false;
    if (theta < (M_PI / 2.0)) {
      reversed = true;
    }
    for (size_t i = 0; i < output.points.size(); ++i) {
      if (reversed) {
        size_t j = output.points.size() - i - 1;
        poly.polygon.points[i].x = output.points[j].x;
        poly.polygon.points[i].y = output.points[j].y;
        poly.polygon.points[i].z = output.points[j].z;
      } else {
        poly.polygon.points[i].x = output.points[i].x;
        poly.polygon.points[i].y = output.points[i].y;
        poly.polygon.points[i].z = output.points[i].z;
      }
    }
    pub_plane_.publish(boost::make_shared<const geometry_msgs::PolygonStamped>(poly));
  }
  // Publish a Boost shared ptr const data
  output.header = cloud->header;
  pub_output_.publish(ros_ptr(output.makeShared()));
}

typedef pcl_ros::ConvexHull2D ConvexHull2D;
PLUGINLIB_EXPORT_CLASS(ConvexHull2D, nodelet::Nodelet)
