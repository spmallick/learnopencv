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
 * $Id: voxel_grid.cpp 35876 2011-02-09 01:04:36Z rusu $
 *
 */

#include "pcl_ros/filters/voxel_grid.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////

pcl_ros::VoxelGrid::VoxelGrid(const rclcpp::NodeOptions & options)
: Filter("VoxelGridNode", options)
{
  std::vector<std::string> common_param_names = add_common_params();

  rcl_interfaces::msg::ParameterDescriptor leaf_size_desc;
  leaf_size_desc.name = "leaf_size";
  leaf_size_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE;
  leaf_size_desc.description =
    "The size of a leaf (on x,y,z) used for downsampling";
  {
    rcl_interfaces::msg::FloatingPointRange float_range;
    float_range.from_value = 0.0;
    float_range.to_value = 1.0;
    leaf_size_desc.floating_point_range.push_back(float_range);
  }
  declare_parameter(leaf_size_desc.name, rclcpp::ParameterValue(0.01), leaf_size_desc);

  rcl_interfaces::msg::ParameterDescriptor min_points_per_voxel_desc;
  min_points_per_voxel_desc.name = "min_points_per_voxel";
  min_points_per_voxel_desc.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
  min_points_per_voxel_desc.description =
    "The minimum number of points required for a voxel to be used.";
  {
    rcl_interfaces::msg::IntegerRange int_range;
    int_range.from_value = 1;
    int_range.to_value = 100000;
    min_points_per_voxel_desc.integer_range.push_back(int_range);
  }
  declare_parameter(
    min_points_per_voxel_desc.name, rclcpp::ParameterValue(2), min_points_per_voxel_desc);

  std::vector<std::string> param_names {
    leaf_size_desc.name,
    min_points_per_voxel_desc.name,
  };
  param_names.insert(param_names.end(), common_param_names.begin(), common_param_names.end());

  callback_handle_ =
    add_on_set_parameters_callback(
    std::bind(
      &VoxelGrid::config_callback, this,
      std::placeholders::_1));

  config_callback(get_parameters(param_names));

  // TODO(daisukes): lazy subscription after rclcpp#2060
  subscribe();
}

void
pcl_ros::VoxelGrid::filter(
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
pcl_ros::VoxelGrid::config_callback(const std::vector<rclcpp::Parameter> & params)
{
  std::lock_guard<std::mutex> lock(mutex_);

  double filter_min, filter_max;
  impl_.getFilterLimits(filter_min, filter_max);

  Eigen::Vector3f leaf_size = impl_.getLeafSize();

  unsigned int minPointsPerVoxel = impl_.getMinimumPointsNumberPerVoxel();

  for (const rclcpp::Parameter & param : params) {
    if (param.get_name() == "filter_field_name") {
      // Check the current value for the filter field
      if (impl_.getFilterFieldName() != param.as_string()) {
        // Set the filter field if different
        impl_.setFilterFieldName(param.as_string());
        RCLCPP_DEBUG(
          get_logger(), "Setting the filter field name to: %s.",
          param.as_string().c_str());
      }
    }
    if (param.get_name() == "filter_limit_min") {
      // Check the current values for filter min-max
      if (filter_min != param.as_double()) {
        filter_min = param.as_double();
        RCLCPP_DEBUG(
          get_logger(),
          "Setting the minimum filtering value a point will be considered from to: %f.",
          filter_min);
        // Set the filter min-max if different
        impl_.setFilterLimits(filter_min, filter_max);
      }
    }
    if (param.get_name() == "filter_limit_max") {
      // Check the current values for filter min-max
      if (filter_max != param.as_double()) {
        filter_max = param.as_double();
        RCLCPP_DEBUG(
          get_logger(),
          "Setting the maximum filtering value a point will be considered from to: %f.",
          filter_max);
        // Set the filter min-max if different
        impl_.setFilterLimits(filter_min, filter_max);
      }
    }
    if (param.get_name() == "filter_limit_negative") {
      bool new_filter_limits_negative = param.as_bool();
      if (impl_.getFilterLimitsNegative() != new_filter_limits_negative) {
        RCLCPP_DEBUG(
          get_logger(),
          "Setting the filter negative flag to: %s.",
          (new_filter_limits_negative ? "true" : "false"));
        impl_.setFilterLimitsNegative(new_filter_limits_negative);
      }
    }
    if (param.get_name() == "min_points_per_voxel") {
      if (minPointsPerVoxel != ((unsigned int) param.as_int())) {
        RCLCPP_DEBUG(
          get_logger(),
          "Setting the minimum points per voxel to: %u.",
          minPointsPerVoxel);
        impl_.setMinimumPointsNumberPerVoxel(param.as_int());
      }
    }
    if (param.get_name() == "leaf_size") {
      leaf_size.setConstant(param.as_double());
      if (impl_.getLeafSize() != leaf_size) {
        RCLCPP_DEBUG(
          get_logger(), "Setting the downsampling leaf size to: %f %f %f.",
          leaf_size[0], leaf_size[1], leaf_size[2]);
        impl_.setLeafSize(leaf_size[0], leaf_size[1], leaf_size[2]);
      }
    }
  }

  // Range constraints are enforced by rclcpp::Parameter.
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  return result;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_ros::VoxelGrid)
