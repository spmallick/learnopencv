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
 * $Id: pointcloud_to_pcd.cpp 33238 2010-03-11 00:46:58Z rusu $
 *
 */

/**
\author Radu Bogdan Rusu

@b pointcloud_to_pcd is a simple node that retrieves a ROS point cloud message and saves it to disk into a PCD (Point
Cloud Data) file format.

**/

#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

namespace pcl_ros
{

class PointCloudToPCD : public rclcpp::Node
{
private:
  std::string prefix_;
  bool binary_;
  bool compressed_;
  bool rgb_;
  bool use_transform_;
  std::string fixed_frame_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

public:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;

  void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    if (cloud_msg->data.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Received empty point cloud message!");
      return;
    }

    Eigen::Vector4f v = Eigen::Vector4f::Zero();
    Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
    if (!fixed_frame_.empty()) {
      use_transform_ = false;
      try {
        geometry_msgs::msg::TransformStamped transform;
        transform = tf_buffer_.lookupTransform(
          fixed_frame_, cloud_msg->header.frame_id,
          cloud_msg->header.stamp);

        v = Eigen::Vector4f::Zero();
        v.head<3>() = Eigen::Vector3f(
          transform.transform.translation.x,
          transform.transform.translation.y,
          transform.transform.translation.z);
        q = Eigen::Quaternionf(
          transform.transform.rotation.w,
          transform.transform.rotation.x,
          transform.transform.rotation.y,
          transform.transform.rotation.z);

        Eigen::Affine3d transform_eigen;
        transform_eigen =
          tf2::transformToEigen(
          tf_buffer_.lookupTransform(
            fixed_frame_, cloud_msg->header.frame_id,
            cloud_msg->header.stamp));
        v = Eigen::Vector4f::Zero();
        v.head<3>() = transform_eigen.translation().cast<float>();
        q = transform_eigen.rotation().cast<float>();
        use_transform_ = true;
      } catch (tf2::LookupException & ex) {
        RCLCPP_WARN(this->get_logger(), "skip transform: %s", ex.what());
      } catch (tf2::TransformException & ex) {
        RCLCPP_ERROR(this->get_logger(), "skip transform: %s", ex.what());
      }
    } else {
      use_transform_ = false;
    }

    std::stringstream ss;
    ss << prefix_ << cloud_msg->header.stamp.sec << "." << cloud_msg->header.stamp.nanosec <<
      ".pcd";
    RCLCPP_INFO(this->get_logger(), "Writing to %s", ss.str().c_str());
    if (rgb_) {
      pcl::PointCloud<pcl::PointXYZRGB> cloud;
      pcl::fromROSMsg(*cloud_msg, cloud);
      if (use_transform_) {
        transformPointCloud(cloud, cloud, v, q);
      }
      writePCDFile(ss.str(), cloud);
    } else {
      pcl::PointCloud<pcl::PointXYZ> cloud;
      pcl::fromROSMsg(*cloud_msg, cloud);
      if (use_transform_) {
        transformPointCloud(cloud, cloud, v, q);
      }
      writePCDFile(ss.str(), cloud);
    }
  }

  template<typename T>
  void transformPointCloud(
    const pcl::PointCloud<T> & cloud_in, pcl::PointCloud<T> & cloud_out,
    const Eigen::Vector4f & v, const Eigen::Quaternionf & q)
  {
    cloud_out = cloud_in;
    for (size_t i = 0; i < cloud_in.size(); ++i) {
      Eigen::Vector3f pt = cloud_in[i].getVector3fMap();
      pt = q * pt + v.head<3>();
      cloud_out[i].x = pt[0];
      cloud_out[i].y = pt[1];
      cloud_out[i].z = pt[2];
    }
  }

  template<typename T>
  void writePCDFile(const std::string & filename, const pcl::PointCloud<T> & cloud)
  {
    pcl::PCDWriter writer;
    if (binary_) {
      if (compressed_) {
        writer.writeBinaryCompressed(filename, cloud);
      } else {
        writer.writeBinary(filename, cloud);
      }
    } else {
      writer.writeASCII(filename, cloud, 8);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  explicit PointCloudToPCD(const rclcpp::NodeOptions & options)
  : rclcpp::Node("pointcloud_to_pcd", options),
    binary_(false), compressed_(false), rgb_(false),
    tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
  {
    this->declare_parameter("prefix", prefix_);
    this->declare_parameter("fixed_frame", fixed_frame_);
    this->declare_parameter("binary", binary_);
    this->declare_parameter("compressed", compressed_);
    this->declare_parameter("rgb", rgb_);

    this->get_parameter("prefix", prefix_);
    this->get_parameter("fixed_frame", fixed_frame_);
    this->get_parameter("binary", binary_);
    this->get_parameter("compressed", compressed_);
    this->get_parameter("rgb", rgb_);

    auto sensor_qos = rclcpp::SensorDataQoS();
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "input", sensor_qos,
      std::bind(&PointCloudToPCD::cloud_cb, this, std::placeholders::_1));
  }
};
}  // namespace pcl_ros

RCLCPP_COMPONENTS_REGISTER_NODE(pcl_ros::PointCloudToPCD)
