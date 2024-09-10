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
 *
 */

#include "pcl_ros/transforms.hpp"
#include "pcl_ros/impl/transforms.hpp"

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/convert.h>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <string>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/time.hpp>

namespace pcl_ros
{
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
transformPointCloud(
  const std::string & target_frame, const sensor_msgs::msg::PointCloud2 & in,
  sensor_msgs::msg::PointCloud2 & out, const tf2_ros::Buffer & tf_buffer)
{
  if (in.header.frame_id == target_frame) {
    out = in;
    return true;
  }

  // Get the TF transform
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform =
      tf_buffer.lookupTransform(
      target_frame, in.header.frame_id, tf2_ros::fromMsg(
        in.header.stamp), tf2::Duration(std::chrono::seconds(1)));
  } catch (tf2::LookupException & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
    return false;
  } catch (tf2::ExtrapolationException & e) {
    RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
    return false;
  }

  // Convert the TF transform to Eigen format
  Eigen::Matrix4f eigen_transform;
  transformAsMatrix(transform, eigen_transform);

  transformPointCloud(eigen_transform, in, out);

  out.header.frame_id = target_frame;
  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
transformPointCloud(
  const std::string & target_frame, const tf2::Transform & net_transform,
  const sensor_msgs::msg::PointCloud2 & in, sensor_msgs::msg::PointCloud2 & out)
{
  if (in.header.frame_id == target_frame) {
    out = in;
    return;
  }

  // Get the transformation
  Eigen::Matrix4f transform;
  transformAsMatrix(net_transform, transform);

  transformPointCloud(transform, in, out);

  out.header.frame_id = target_frame;
}

void
transformPointCloud(
  const std::string & target_frame, const geometry_msgs::msg::TransformStamped & net_transform,
  const sensor_msgs::msg::PointCloud2 & in, sensor_msgs::msg::PointCloud2 & out)
{
  tf2::Transform transform;
  tf2::convert(net_transform.transform, transform);
  transformPointCloud(target_frame, transform, in, out);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
transformPointCloud(
  const Eigen::Matrix4f & transform, const sensor_msgs::msg::PointCloud2 & in,
  sensor_msgs::msg::PointCloud2 & out)
{
  // Get X-Y-Z indices
  int x_idx = pcl::getFieldIndex(in, "x");
  int y_idx = pcl::getFieldIndex(in, "y");
  int z_idx = pcl::getFieldIndex(in, "z");

  if (x_idx == -1 || y_idx == -1 || z_idx == -1) {
    RCLCPP_ERROR(
      rclcpp::get_logger("pcl_ros"),
      "Input dataset has no X-Y-Z coordinates! Cannot convert to Eigen format.");
    return;
  }

  if (in.fields[x_idx].datatype != sensor_msgs::msg::PointField::FLOAT32 ||
    in.fields[y_idx].datatype != sensor_msgs::msg::PointField::FLOAT32 ||
    in.fields[z_idx].datatype != sensor_msgs::msg::PointField::FLOAT32)
  {
    RCLCPP_ERROR(
      rclcpp::get_logger("pcl_ros"),
      "X-Y-Z coordinates not floats. Currently only floats are supported.");
    return;
  }

  // Check if distance is available
  int dist_idx = pcl::getFieldIndex(in, "distance");

  // Copy the other data
  if (&in != &out) {
    out.header = in.header;
    out.height = in.height;
    out.width = in.width;
    out.fields = in.fields;
    out.is_bigendian = in.is_bigendian;
    out.point_step = in.point_step;
    out.row_step = in.row_step;
    out.is_dense = in.is_dense;
    out.data.resize(in.data.size());
    // Copy everything as it's faster than copying individual elements
    memcpy(out.data.data(), in.data.data(), in.data.size());
  }

  Eigen::Array4i xyz_offset(in.fields[x_idx].offset, in.fields[y_idx].offset,
    in.fields[z_idx].offset, 0);

  for (size_t i = 0; i < in.width * in.height; ++i) {
    Eigen::Vector4f pt(*reinterpret_cast<const float *>(&in.data[xyz_offset[0]]),
      *reinterpret_cast<const float *>(&in.data[xyz_offset[1]]),
      *reinterpret_cast<const float *>(&in.data[xyz_offset[2]]), 1);
    Eigen::Vector4f pt_out;

    bool max_range_point = false;
    int distance_ptr_offset = i * in.point_step + in.fields[dist_idx].offset;
    const float * distance_ptr = (dist_idx < 0 ?
      NULL : reinterpret_cast<const float *>(&in.data[distance_ptr_offset]));
    if (!std::isfinite(pt[0]) || !std::isfinite(pt[1]) || !std::isfinite(pt[2])) {
      if (distance_ptr == NULL || !std::isfinite(*distance_ptr)) {  // Invalid point
        pt_out = pt;
      } else {  // max range point
        pt[0] = *distance_ptr;  // Replace x with the x value saved in distance
        pt_out = transform * pt;
        max_range_point = true;
      }
    } else {
      pt_out = transform * pt;
    }

    if (max_range_point) {
      // Save x value in distance again
      *reinterpret_cast<float *>(&out.data[distance_ptr_offset]) = pt_out[0];
      pt_out[0] = std::numeric_limits<float>::quiet_NaN();
    }

    memcpy(&out.data[xyz_offset[0]], &pt_out[0], sizeof(float));
    memcpy(&out.data[xyz_offset[1]], &pt_out[1], sizeof(float));
    memcpy(&out.data[xyz_offset[2]], &pt_out[2], sizeof(float));


    xyz_offset += in.point_step;
  }

  // Check if the viewpoint information is present
  int vp_idx = pcl::getFieldIndex(in, "vp_x");
  if (vp_idx != -1) {
    // Transform the viewpoint info too
    for (size_t i = 0; i < out.width * out.height; ++i) {
      float * pstep =
        reinterpret_cast<float *>(&out.data[i * out.point_step + out.fields[vp_idx].offset]);
      // Assume vp_x, vp_y, vp_z are consecutive
      Eigen::Vector4f vp_in(pstep[0], pstep[1], pstep[2], 1);
      Eigen::Vector4f vp_out = transform * vp_in;

      pstep[0] = vp_out[0];
      pstep[1] = vp_out[1];
      pstep[2] = vp_out[2];
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
transformAsMatrix(const tf2::Transform & bt, Eigen::Matrix4f & out_mat)
{
  double mv[12];
  bt.getBasis().getOpenGLSubMatrix(mv);

  tf2::Vector3 origin = bt.getOrigin();

  out_mat(0, 0) = mv[0]; out_mat(0, 1) = mv[4]; out_mat(0, 2) = mv[8];
  out_mat(1, 0) = mv[1]; out_mat(1, 1) = mv[5]; out_mat(1, 2) = mv[9];
  out_mat(2, 0) = mv[2]; out_mat(2, 1) = mv[6]; out_mat(2, 2) = mv[10];

  out_mat(3, 0) = out_mat(3, 1) = out_mat(3, 2) = 0; out_mat(3, 3) = 1;
  out_mat(0, 3) = origin.x();
  out_mat(1, 3) = origin.y();
  out_mat(2, 3) = origin.z();
}

void
transformAsMatrix(const geometry_msgs::msg::TransformStamped & bt, Eigen::Matrix4f & out_mat)
{
  tf2::Transform transform;
  tf2::convert(bt.transform, transform);
  transformAsMatrix(transform, out_mat);
}
}  // namespace pcl_ros

//////////////////////////////////////////////////////////////////////////////////////////////
template void pcl_ros::transformPointCloudWithNormals<pcl::PointNormal>(
  const pcl::PointCloud<pcl::PointNormal> &, pcl::PointCloud<pcl::PointNormal> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloudWithNormals<pcl::PointXYZRGBNormal>(
  const pcl::PointCloud<pcl::PointXYZRGBNormal> &, pcl::PointCloud<pcl::PointXYZRGBNormal> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloudWithNormals<pcl::PointXYZINormal>(
  const pcl::PointCloud<pcl::PointXYZINormal> &, pcl::PointCloud<pcl::PointXYZINormal> &,
  const tf2::Transform &);

//////////////////////////////////////////////////////////////////////////////////////////////
template void pcl_ros::transformPointCloudWithNormals<pcl::PointNormal>(
  const pcl::PointCloud<pcl::PointNormal> &, pcl::PointCloud<pcl::PointNormal> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloudWithNormals<pcl::PointXYZRGBNormal>(
  const pcl::PointCloud<pcl::PointXYZRGBNormal> &, pcl::PointCloud<pcl::PointXYZRGBNormal> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloudWithNormals<pcl::PointXYZINormal>(
  const pcl::PointCloud<pcl::PointXYZINormal> &, pcl::PointCloud<pcl::PointXYZINormal> &,
  const geometry_msgs::msg::TransformStamped &);

//////////////////////////////////////////////////////////////////////////////////////////////
template bool pcl_ros::transformPointCloudWithNormals<pcl::PointNormal>(
  const std::string &,
  const pcl::PointCloud<pcl::PointNormal> &, pcl::PointCloud<pcl::PointNormal> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloudWithNormals<pcl::PointXYZRGBNormal>(
  const std::string &,
  const pcl::PointCloud<pcl::PointXYZRGBNormal> &, pcl::PointCloud<pcl::PointXYZRGBNormal> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloudWithNormals<pcl::PointXYZINormal>(
  const std::string &,
  const pcl::PointCloud<pcl::PointXYZINormal> &, pcl::PointCloud<pcl::PointXYZINormal> &,
  const tf2_ros::Buffer &);

//////////////////////////////////////////////////////////////////////////////////////////////
template bool pcl_ros::transformPointCloudWithNormals<pcl::PointNormal>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointNormal> &, const std::string &,
  pcl::PointCloud<pcl::PointNormal> &, const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloudWithNormals<pcl::PointXYZRGBNormal>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointXYZRGBNormal> &, const std::string &,
  pcl::PointCloud<pcl::PointXYZRGBNormal> &, const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloudWithNormals<pcl::PointXYZINormal>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointXYZINormal> &, const std::string &,
  pcl::PointCloud<pcl::PointXYZINormal> &, const tf2_ros::Buffer &);

//////////////////////////////////////////////////////////////////////////////////////////////
template void pcl_ros::transformPointCloud<pcl::PointXYZ>(
  const pcl::PointCloud<pcl::PointXYZ> &,
  pcl::PointCloud<pcl::PointXYZ> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloud<pcl::PointXYZI>(
  const pcl::PointCloud<pcl::PointXYZI> &,
  pcl::PointCloud<pcl::PointXYZI> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloud<pcl::PointXYZRGBA>(
  const pcl::PointCloud<pcl::PointXYZRGBA> &, pcl::PointCloud<pcl::PointXYZRGBA> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloud<pcl::PointXYZRGB>(
  const pcl::PointCloud<pcl::PointXYZRGB> &, pcl::PointCloud<pcl::PointXYZRGB> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloud<pcl::InterestPoint>(
  const pcl::PointCloud<pcl::InterestPoint> &, pcl::PointCloud<pcl::InterestPoint> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloud<pcl::PointNormal>(
  const pcl::PointCloud<pcl::PointNormal> &, pcl::PointCloud<pcl::PointNormal> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloud<pcl::PointXYZRGBNormal>(
  const pcl::PointCloud<pcl::PointXYZRGBNormal> &, pcl::PointCloud<pcl::PointXYZRGBNormal> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloud<pcl::PointXYZINormal>(
  const pcl::PointCloud<pcl::PointXYZINormal> &, pcl::PointCloud<pcl::PointXYZINormal> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloud<pcl::PointWithRange>(
  const pcl::PointCloud<pcl::PointWithRange> &, pcl::PointCloud<pcl::PointWithRange> &,
  const tf2::Transform &);
template void pcl_ros::transformPointCloud<pcl::PointWithViewpoint>(
  const pcl::PointCloud<pcl::PointWithViewpoint> &, pcl::PointCloud<pcl::PointWithViewpoint> &,
  const tf2::Transform &);

//////////////////////////////////////////////////////////////////////////////////////////////
template void pcl_ros::transformPointCloud<pcl::PointXYZ>(
  const pcl::PointCloud<pcl::PointXYZ> &,
  pcl::PointCloud<pcl::PointXYZ> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloud<pcl::PointXYZI>(
  const pcl::PointCloud<pcl::PointXYZI> &,
  pcl::PointCloud<pcl::PointXYZI> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloud<pcl::PointXYZRGBA>(
  const pcl::PointCloud<pcl::PointXYZRGBA> &, pcl::PointCloud<pcl::PointXYZRGBA> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloud<pcl::PointXYZRGB>(
  const pcl::PointCloud<pcl::PointXYZRGB> &, pcl::PointCloud<pcl::PointXYZRGB> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloud<pcl::InterestPoint>(
  const pcl::PointCloud<pcl::InterestPoint> &, pcl::PointCloud<pcl::InterestPoint> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloud<pcl::PointNormal>(
  const pcl::PointCloud<pcl::PointNormal> &, pcl::PointCloud<pcl::PointNormal> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloud<pcl::PointXYZRGBNormal>(
  const pcl::PointCloud<pcl::PointXYZRGBNormal> &, pcl::PointCloud<pcl::PointXYZRGBNormal> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloud<pcl::PointXYZINormal>(
  const pcl::PointCloud<pcl::PointXYZINormal> &, pcl::PointCloud<pcl::PointXYZINormal> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloud<pcl::PointWithRange>(
  const pcl::PointCloud<pcl::PointWithRange> &, pcl::PointCloud<pcl::PointWithRange> &,
  const geometry_msgs::msg::TransformStamped &);
template void pcl_ros::transformPointCloud<pcl::PointWithViewpoint>(
  const pcl::PointCloud<pcl::PointWithViewpoint> &, pcl::PointCloud<pcl::PointWithViewpoint> &,
  const geometry_msgs::msg::TransformStamped &);

//////////////////////////////////////////////////////////////////////////////////////////////
template bool pcl_ros::transformPointCloud<pcl::PointXYZ>(
  const std::string &,
  const pcl::PointCloud<pcl::PointXYZ> &,
  pcl::PointCloud<pcl::PointXYZ> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZI>(
  const std::string &,
  const pcl::PointCloud<pcl::PointXYZI> &,
  pcl::PointCloud<pcl::PointXYZI> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZRGBA>(
  const std::string &,
  const pcl::PointCloud<pcl::PointXYZRGBA> &, pcl::PointCloud<pcl::PointXYZRGBA> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZRGB>(
  const std::string &,
  const pcl::PointCloud<pcl::PointXYZRGB> &, pcl::PointCloud<pcl::PointXYZRGB> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::InterestPoint>(
  const std::string &,
  const pcl::PointCloud<pcl::InterestPoint> &, pcl::PointCloud<pcl::InterestPoint> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointNormal>(
  const std::string &,
  const pcl::PointCloud<pcl::PointNormal> &, pcl::PointCloud<pcl::PointNormal> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZRGBNormal>(
  const std::string &,
  const pcl::PointCloud<pcl::PointXYZRGBNormal> &, pcl::PointCloud<pcl::PointXYZRGBNormal> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZINormal>(
  const std::string &,
  const pcl::PointCloud<pcl::PointXYZINormal> &, pcl::PointCloud<pcl::PointXYZINormal> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointWithRange>(
  const std::string &,
  const pcl::PointCloud<pcl::PointWithRange> &, pcl::PointCloud<pcl::PointWithRange> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointWithViewpoint>(
  const std::string &,
  const pcl::PointCloud<pcl::PointWithViewpoint> &, pcl::PointCloud<pcl::PointWithViewpoint> &,
  const tf2_ros::Buffer &);

//////////////////////////////////////////////////////////////////////////////////////////////
template bool pcl_ros::transformPointCloud<pcl::PointXYZ>(
  const std::string &, const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointXYZ> &,
  const std::string &,
  pcl::PointCloud<pcl::PointXYZ> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZI>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointXYZI> &,
  const std::string &,
  pcl::PointCloud<pcl::PointXYZI> &,
  const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZRGBA>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointXYZRGBA> &, const std::string &,
  pcl::PointCloud<pcl::PointXYZRGBA> &, const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZRGB>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointXYZRGB> &, const std::string &,
  pcl::PointCloud<pcl::PointXYZRGB> &, const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::InterestPoint>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::InterestPoint> &, const std::string &,
  pcl::PointCloud<pcl::InterestPoint> &, const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointNormal>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointNormal> &, const std::string &,
  pcl::PointCloud<pcl::PointNormal> &, const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZRGBNormal>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointXYZRGBNormal> &, const std::string &,
  pcl::PointCloud<pcl::PointXYZRGBNormal> &, const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointXYZINormal>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointXYZINormal> &, const std::string &,
  pcl::PointCloud<pcl::PointXYZINormal> &, const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointWithRange>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointWithRange> &, const std::string &,
  pcl::PointCloud<pcl::PointWithRange> &, const tf2_ros::Buffer &);
template bool pcl_ros::transformPointCloud<pcl::PointWithViewpoint>(
  const std::string &,
  const rclcpp::Time &,
  const pcl::PointCloud<pcl::PointWithViewpoint> &, const std::string &,
  pcl::PointCloud<pcl::PointWithViewpoint> &, const tf2_ros::Buffer &);
