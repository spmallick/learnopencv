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
 * $Id: statistical_outlier_removal.cpp 35876 2011-02-09 01:04:36Z rusu $
 *
 */

#include "pcl_ros/filters/statistical_outlier_removal.hpp"

pcl_ros::StatisticalOutlierRemoval::StatisticalOutlierRemoval(const rclcpp::NodeOptions & options)
: Filter("StatisticalOutlierRemovalNode", options)
{
  rcl_interfaces::msg::ParameterDescriptor mean_k_desc;
  mean_k_desc.name = "mean_k";
  mean_k_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
  mean_k_desc.description =
    "The number of points (k) to use for mean distance estimation.";
  {
    rcl_interfaces::msg::IntegerRange int_range;
    int_range.from_value = 2;
    int_range.to_value = 100;
    mean_k_desc.integer_range.push_back(int_range);
  }
  declare_parameter(mean_k_desc.name, rclcpp::ParameterValue(2), mean_k_desc);

  rcl_interfaces::msg::ParameterDescriptor stddev_desc;
  stddev_desc.name = "stddev";
  stddev_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;
  stddev_desc.description =
    "The standard deviation multiplier threshold."
    "All points outside the mean +- sigma * std_mul will be considered outliers.";
  {
    rcl_interfaces::msg::FloatingPointRange float_range;
    float_range.from_value = 0.0;
    float_range.to_value = 5.0;
    stddev_desc.floating_point_range.push_back(float_range);
  }
  declare_parameter(stddev_desc.name, rclcpp::ParameterValue(0.0), stddev_desc);

  rcl_interfaces::msg::ParameterDescriptor negative_desc;
  negative_desc.name = "negative";
  negative_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
  negative_desc.description =
    "Set whether the inliers should be returned (true) or the outliers (false).";
  declare_parameter(negative_desc.name, rclcpp::ParameterValue(false), negative_desc);

  const std::vector<std::string> param_names {
    mean_k_desc.name,
    stddev_desc.name,
    negative_desc.name,
  };

  callback_handle_ =
    add_on_set_parameters_callback(
    std::bind(
      &StatisticalOutlierRemoval::config_callback, this,
      std::placeholders::_1));

  config_callback(get_parameters(param_names));

  // TODO(daisukes): lazy subscription after rclcpp#2060
  subscribe();
}

void
pcl_ros::StatisticalOutlierRemoval::filter(
  const PointCloud2::ConstSharedPtr & input, const IndicesPtr & indices,
  PointCloud2 & output)
{
  std::lock_guard<std::mutex> lock(mutex_);
  pcl::PCLPointCloud2::Ptr pcl_input(new pcl::PCLPointCloud2);
  pcl_conversions::toPCL(*(input), *(pcl_input));
  impl_.setInputCloud(pcl_input);
  impl_.setIndices(indices);
  pcl::PCLPointCloud2 pcl_output;
  impl_.filter(pcl_output);
  pcl_conversions::moveFromPCL(pcl_output, output);
}

//////////////////////////////////////////////////////////////////////////////////////////////
rcl_interfaces::msg::SetParametersResult
pcl_ros::StatisticalOutlierRemoval::config_callback(const std::vector<rclcpp::Parameter> & params)
{
  std::lock_guard<std::mutex> lock(mutex_);

  for (const rclcpp::Parameter & param : params) {
    if (param.get_name() == "mean_k") {
      if (impl_.getMeanK() != param.as_int()) {
        RCLCPP_DEBUG(
          get_logger(),
          "Setting the number of points (k) to use for mean distance estimation to: %ld.",
          param.as_int());
        impl_.setMeanK(param.as_int());
      }
    }
    if (param.get_name() == "stddev") {
      if (impl_.getStddevMulThresh() != param.as_double()) {
        RCLCPP_DEBUG(
          get_logger(),
          "Setting the standard deviation multiplier threshold to: %f.",
          param.as_double());
        impl_.setStddevMulThresh(param.as_double());
      }
    }
    if (param.get_name() == "negative") {
      if (impl_.getNegative() != param.as_bool()) {
        RCLCPP_DEBUG(
          get_logger(),
          "Returning only inliers: %s.",
          (param.as_bool() ? "false" : "true"));
        impl_.setNegative(param.as_bool());
      }
    }
  }

  // Range constraints are enforced by rclcpp::Parameter.
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  return result;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_ros::StatisticalOutlierRemoval)
