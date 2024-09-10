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
 * $Id: moving_least_squares.cpp 36097 2011-02-20 14:18:58Z marton $
 *
 */

#include <pluginlib/class_list_macros.h>
#include <pcl/common/io.h>
#include <vector>
#include "pcl_ros/surface/moving_least_squares.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::MovingLeastSquares::onInit()
{
  PCLNodelet::onInit();

  // ros::NodeHandle private_nh = getMTPrivateNodeHandle ();
  pub_output_ = advertise<PointCloudIn>(*pnh_, "output", max_queue_size_);
  pub_normals_ = advertise<NormalCloudOut>(*pnh_, "normals", max_queue_size_);

  // if (!pnh_->getParam ("k_search", k_) && !pnh_->getParam ("search_radius", search_radius_))
  if (!pnh_->getParam("search_radius", search_radius_)) {
    // NODELET_ERROR ("[%s::onInit] Neither 'k_search' nor 'search_radius' set! Need to set "
    // "at least one of these parameters before continuing.", getName ().c_str ());
    NODELET_ERROR(
      "[%s::onInit] Need a 'search_radius' parameter to be set before continuing!",
      getName().c_str());
    return;
  }
  if (!pnh_->getParam("spatial_locator", spatial_locator_type_)) {
    NODELET_ERROR(
      "[%s::onInit] Need a 'spatial_locator' parameter to be set before continuing!",
      getName().c_str());
    return;
  }

  // Enable the dynamic reconfigure service
  srv_ = boost::make_shared<dynamic_reconfigure::Server<MLSConfig>>(*pnh_);
  dynamic_reconfigure::Server<MLSConfig>::CallbackType f = boost::bind(
    &MovingLeastSquares::config_callback, this, _1, _2);
  srv_->setCallback(f);

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
pcl_ros::MovingLeastSquares::subscribe()
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
            message_filters::sync_policies::ApproximateTime<
              PointCloudIn, PointIndices>>>(max_queue_size_);
      // surface not enabled, connect the input-indices duo and register
      sync_input_indices_a_->connectInput(sub_input_filter_, sub_indices_filter_);
      sync_input_indices_a_->registerCallback(
        bind(
          &MovingLeastSquares::input_indices_callback,
          this, _1, _2));
    } else {
      sync_input_indices_e_ =
        boost::make_shared<message_filters::Synchronizer<
            message_filters::sync_policies::ExactTime<PointCloudIn,
            PointIndices>>>(max_queue_size_);
      // surface not enabled, connect the input-indices duo and register
      sync_input_indices_e_->connectInput(sub_input_filter_, sub_indices_filter_);
      sync_input_indices_e_->registerCallback(
        bind(
          &MovingLeastSquares::input_indices_callback,
          this, _1, _2));
    }
  } else {
    // Subscribe in an old fashion to input only (no filters)
    sub_input_ =
      pnh_->subscribe<PointCloudIn>(
      "input", 1,
      bind(&MovingLeastSquares::input_indices_callback, this, _1, PointIndicesConstPtr()));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::MovingLeastSquares::unsubscribe()
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
pcl_ros::MovingLeastSquares::input_indices_callback(
  const PointCloudInConstPtr & cloud,
  const PointIndicesConstPtr & indices)
{
  // No subscribers, no work
  if (pub_output_.getNumSubscribers() <= 0 && pub_normals_.getNumSubscribers() <= 0) {
    return;
  }

  // Output points have the same type as the input, they are only smoothed
  PointCloudIn output;

  // Normals are also estimated and published on a separate topic
  NormalCloudOut::Ptr normals(new NormalCloudOut());

  // If cloud is given, check if it's valid
  if (!isValid(cloud)) {
    NODELET_ERROR("[%s::input_indices_callback] Invalid input!", getName().c_str());
    output.header = cloud->header;
    pub_output_.publish(ros_ptr(output.makeShared()));
    return;
  }
  // If indices are given, check if they are valid
  if (indices && !isValid(indices, "indices")) {
    NODELET_ERROR("[%s::input_indices_callback] Invalid indices!", getName().c_str());
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
      "[%s::input_callback] PointCloud with %d data points, stamp %f, and frame %s on "
      "topic %s received.",
      getName().c_str(), cloud->width * cloud->height, fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(),
      getMTPrivateNodeHandle().resolveName("input").c_str());
  }
  ///

  // Reset the indices and surface pointers
  impl_.setInputCloud(pcl_ptr(cloud));

  IndicesPtr indices_ptr;
  if (indices) {
    indices_ptr.reset(new std::vector<int>(indices->indices));
  }

  impl_.setIndices(indices_ptr);

  // Initialize the spatial locator

  // Do the reconstructon
  // impl_.process (output);

  // Publish a Boost shared ptr const data
  // Enforce that the TF frame and the timestamp are copied
  output.header = cloud->header;
  pub_output_.publish(ros_ptr(output.makeShared()));
  normals->header = cloud->header;
  pub_normals_.publish(ros_ptr(normals));
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::MovingLeastSquares::config_callback(MLSConfig & config, uint32_t level)
{
  // \Note Zoli, shouldn't this be implemented in MLS too?
  /*if (k_ != config.k_search)
  {
    k_ = config.k_search;
    NODELET_DEBUG ("[config_callback] Setting the k_search to: %d.", k_);
  }*/
  if (search_radius_ != config.search_radius) {
    search_radius_ = config.search_radius;
    NODELET_DEBUG("[config_callback] Setting the search radius: %f.", search_radius_);
    impl_.setSearchRadius(search_radius_);
  }
  if (spatial_locator_type_ != config.spatial_locator) {
    spatial_locator_type_ = config.spatial_locator;
    NODELET_DEBUG(
      "[config_callback] Setting the spatial locator to type: %d.",
      spatial_locator_type_);
  }
  if (use_polynomial_fit_ != config.use_polynomial_fit) {
    use_polynomial_fit_ = config.use_polynomial_fit;
    NODELET_DEBUG(
      "[config_callback] Setting the use_polynomial_fit flag to: %d.",
      use_polynomial_fit_);
#if PCL_VERSION_COMPARE(<, 1, 9, 0)
    impl_.setPolynomialFit(use_polynomial_fit_);
#else
    if (use_polynomial_fit_) {
      NODELET_WARN(
        "[config_callback] use_polynomial_fit is deprecated, use polynomial_order instead!");
      if (impl_.getPolynomialOrder() < 2) {
        impl_.setPolynomialOrder(2);
      }
    } else {
      impl_.setPolynomialOrder(0);
    }
#endif
  }
  if (polynomial_order_ != config.polynomial_order) {
    polynomial_order_ = config.polynomial_order;
    NODELET_DEBUG("[config_callback] Setting the polynomial order to: %d.", polynomial_order_);
    impl_.setPolynomialOrder(polynomial_order_);
  }
  if (gaussian_parameter_ != config.gaussian_parameter) {
    gaussian_parameter_ = config.gaussian_parameter;
    NODELET_DEBUG("[config_callback] Setting the gaussian parameter to: %f.", gaussian_parameter_);
    impl_.setSqrGaussParam(gaussian_parameter_ * gaussian_parameter_);
  }
}

typedef pcl_ros::MovingLeastSquares MovingLeastSquares;
PLUGINLIB_EXPORT_CLASS(MovingLeastSquares, nodelet::Nodelet)
