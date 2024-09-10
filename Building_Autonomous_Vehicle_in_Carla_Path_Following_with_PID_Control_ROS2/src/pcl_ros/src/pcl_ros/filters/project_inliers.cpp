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
 * $Id: project_inliers.cpp 35876 2011-02-09 01:04:36Z rusu $
 *
 */

#include "pcl_ros/filters/project_inliers.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////
pcl_ros::ProjectInliers::ProjectInliers(const rclcpp::NodeOptions & options)
: Filter("ProjectInliersNode", options), model_()
{
  // ---[ Mandatory parameters
  // The type of model to use (user given parameter).
  declare_parameter("model_type", rclcpp::ParameterType::PARAMETER_INTEGER);
  int model_type;
  if (!get_parameter("model_type", model_type)) {
    RCLCPP_ERROR(
      get_logger(),
      "[onConstruct] Need a 'model_type' parameter to be set before continuing!");
    return;
  }
  // ---[ Optional parameters
  // True if all data will be returned, false if only the projected inliers. Default: false.
  declare_parameter("copy_all_data", rclcpp::ParameterValue(false));
  bool copy_all_data = get_parameter("copy_all_data").as_bool();

  // True if all fields will be returned, false if only XYZ. Default: true.
  declare_parameter("copy_all_fields", rclcpp::ParameterValue(true));
  bool copy_all_fields = get_parameter("copy_all_fields").as_bool();

  pub_output_ = create_publisher<PointCloud2>("output", max_queue_size_);

  RCLCPP_DEBUG(
    this->get_logger(),
    "[onConstruct] Node successfully created with the following parameters:\n"
    "  - model_type      : %d\n"
    "  - copy_all_data   : %s\n"
    "  - copy_all_fields : %s",
    model_type, (copy_all_data) ? "true" : "false", (copy_all_fields) ? "true" : "false");

  // Set given parameters here
  impl_.setModelType(model_type);
  impl_.setCopyAllFields(copy_all_fields);
  impl_.setCopyAllData(copy_all_data);

  // TODO(daisukes): lazy subscription after rclcpp#2060
  subscribe();
}

void
pcl_ros::ProjectInliers::filter(
  const PointCloud2::ConstSharedPtr & input, const IndicesPtr & indices,
  PointCloud2 & output)
{
  pcl::PCLPointCloud2::Ptr pcl_input(new pcl::PCLPointCloud2);
  pcl_conversions::toPCL(*(input), *(pcl_input));
  impl_.setInputCloud(pcl_input);
  impl_.setIndices(indices);
  pcl::ModelCoefficients::Ptr pcl_model(new pcl::ModelCoefficients);
  pcl_conversions::toPCL(*(model_), *(pcl_model));
  impl_.setModelCoefficients(pcl_model);
  pcl::PCLPointCloud2 pcl_output;
  impl_.filter(pcl_output);
  pcl_conversions::moveFromPCL(pcl_output, output);
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ProjectInliers::subscribe()
{
  RCLCPP_DEBUG(get_logger(), "subscribe");
/*
  TODO : implement use_indices_
  if (use_indices_)
  {*/

  auto qos_profile = rclcpp::QoS(
    rclcpp::KeepLast(max_queue_size_),
    rmw_qos_profile_default).get_rmw_qos_profile();
  auto sensor_qos_profile = rclcpp::QoS(
    rclcpp::KeepLast(max_queue_size_),
    rmw_qos_profile_sensor_data).get_rmw_qos_profile();
  sub_input_filter_.subscribe(this, "input", sensor_qos_profile);
  sub_indices_filter_.subscribe(this, "indices", qos_profile);
  sub_model_.subscribe(this, "model", qos_profile);

  if (approximate_sync_) {
    sync_input_indices_model_a_ = std::make_shared<
      message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
          PointCloud2, PointIndices, ModelCoefficients>>>(max_queue_size_);
    sync_input_indices_model_a_->connectInput(sub_input_filter_, sub_indices_filter_, sub_model_);
    sync_input_indices_model_a_->registerCallback(
      std::bind(
        &ProjectInliers::input_indices_model_callback, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  } else {
    sync_input_indices_model_e_ = std::make_shared<
      message_filters::Synchronizer<
        message_filters::sync_policies::ExactTime<
          PointCloud2, PointIndices, ModelCoefficients>>>(max_queue_size_);
    sync_input_indices_model_e_->connectInput(sub_input_filter_, sub_indices_filter_, sub_model_);
    sync_input_indices_model_e_->registerCallback(
      std::bind(
        &ProjectInliers::input_indices_model_callback, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ProjectInliers::unsubscribe()
{
/*
  TODO : implement use_indices_
  if (use_indices_)
  {*/
  sub_input_filter_.unsubscribe();
  sub_indices_filter_.unsubscribe();
  sub_model_.unsubscribe();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::ProjectInliers::input_indices_model_callback(
  const PointCloud2::ConstSharedPtr & cloud,
  const PointIndicesConstPtr & indices,
  const ModelCoefficientsConstPtr & model)
{
  if (pub_output_->get_subscription_count() == 0) {
    return;
  }

  if (!isValid(model) || !isValid(indices) || !isValid(cloud)) {
    RCLCPP_ERROR(
      this->get_logger(), "[%s::input_indices_model_callback] Invalid input!", this->get_name());
    return;
  }

  RCLCPP_DEBUG(
    this->get_logger(),
    "[%s::input_indices_model_callback]\n"
    "  - PointCloud with %d data points (%s), stamp %d.%09d, and frame %s on topic %s received.\n"
    "  - PointIndices with %zu values, stamp %d.%09d, and frame %s on topic %s received.\n"
    "  - ModelCoefficients with %zu values, stamp %d.%09d, and frame %s on topic %s received.",
    this->get_name(), cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(),
    cloud->header.stamp.sec, cloud->header.stamp.nanosec, cloud->header.frame_id.c_str(), "input",
    indices->indices.size(), indices->header.stamp.sec, indices->header.stamp.nanosec,
    indices->header.frame_id.c_str(), "inliers", model->values.size(),
    model->header.stamp.sec, model->header.stamp.nanosec, model->header.frame_id.c_str(), "model");

  tf_input_orig_frame_ = cloud->header.frame_id;

  IndicesPtr vindices;
  if (indices) {
    vindices.reset(new std::vector<int>(indices->indices));
  }

  model_ = model;
  computePublish(cloud, vindices);
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_ros::ProjectInliers)
