/*
 * Copyright (c) 2022  Carnegie Mellon University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Carnegie Mellon University nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_msgs/msg/point_indices.hpp>
#include <pcl_msgs/msg/model_coefficients.hpp>

using namespace std::chrono_literals;

namespace pcl_ros_tests_filters
{
class DummyTopics : public rclcpp::Node
{
public:
  explicit DummyTopics(const rclcpp::NodeOptions & options);

private:
  void timer_callback();
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud2_pub_;
  rclcpp::Publisher<pcl_msgs::msg::PointIndices>::SharedPtr indices_pub_;
  rclcpp::Publisher<pcl_msgs::msg::ModelCoefficients>::SharedPtr model_pub_;
  size_t count_;
};

DummyTopics::DummyTopics(const rclcpp::NodeOptions & options)
: Node("dummy_point_cloud2_publisher", options), count_(0)
{
  point_cloud2_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud2", 10);
  indices_pub_ = this->create_publisher<pcl_msgs::msg::PointIndices>("indices", 10);
  model_pub_ = this->create_publisher<pcl_msgs::msg::ModelCoefficients>("model", 10);
  timer_ = this->create_wall_timer(
    100ms, std::bind(&DummyTopics::timer_callback, this));
}

void DummyTopics::timer_callback()
{
  builtin_interfaces::msg::Time now_msg;
  now_msg = get_clock()->now();

  sensor_msgs::msg::PointCloud2 point_cloud2_msg;

  pcl::PointCloud<pcl::PointXYZ> random_pcl;

  unsigned int global_seed = 100;
  for (int v = 0; v < 1000; ++v) {
    pcl::PointXYZ newPoint;
    newPoint.x = (rand_r(&global_seed) * 100.0) / RAND_MAX;
    newPoint.y = (rand_r(&global_seed) * 100.0) / RAND_MAX;
    newPoint.z = (rand_r(&global_seed) * 100.0) / RAND_MAX;
    random_pcl.points.push_back(newPoint);
  }

  // publish point cloud
  pcl::toROSMsg<pcl::PointXYZ>(random_pcl, point_cloud2_msg);
  point_cloud2_msg.header.stamp = now_msg;
  point_cloud2_pub_->publish(point_cloud2_msg);

  // publish indices
  pcl_msgs::msg::PointIndices indices_msg;
  indices_msg.header.stamp = now_msg;
  indices_msg.indices.push_back(0);
  indices_pub_->publish(indices_msg);

  // publish model
  pcl_msgs::msg::ModelCoefficients model_msg;
  model_msg.header.stamp = now_msg;
  model_msg.values.push_back(0);
  model_msg.values.push_back(0);
  model_msg.values.push_back(1);
  model_msg.values.push_back(0);
  model_pub_->publish(model_msg);
}

}  //  namespace pcl_ros_tests_filters

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pcl_ros_tests_filters::DummyTopics)
