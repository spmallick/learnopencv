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
 * $Id: publisher.h 33238 2010-03-11 00:46:58Z rusu $
 *
 */
/**

\author Patrick Mihelich

@b Publisher represents a ROS publisher for the templated PointCloud implementation.

**/

#ifndef PCL_ROS__PUBLISHER_HPP_
#define PCL_ROS__PUBLISHER_HPP_

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/conversions.h>

#include <pcl_conversions/pcl_conversions.h>

#include <string>

namespace pcl_ros
{
class BasePublisher
{
public:
  void
  advertise(ros::NodeHandle & nh, const std::string & topic, uint32_t queue_size)
  {
    pub_ = nh.advertise<sensor_msgs::PointCloud2>(topic, queue_size);
  }

  std::string
  getTopic()
  {
    return pub_.getTopic();
  }

  uint32_t
  getNumSubscribers() const
  {
    return pub_.getNumSubscribers();
  }

  void
  shutdown()
  {
    pub_.shutdown();
  }

  operator void *() const
  {
    return (pub_) ? reinterpret_cast<void *>(1) : reinterpret_cast<void *>(0);
  }

protected:
  ros::Publisher pub_;
};

template<typename PointT>
class Publisher : public BasePublisher
{
public:
  Publisher() {}

  Publisher(ros::NodeHandle & nh, const std::string & topic, uint32_t queue_size)
  {
    advertise(nh, topic, queue_size);
  }

  ~Publisher() {}

  inline void
  publish(const boost::shared_ptr<const pcl::PointCloud<PointT>> & point_cloud) const
  {
    publish(*point_cloud);
  }

  inline void
  publish(const pcl::PointCloud<PointT> & point_cloud) const
  {
    // Fill point cloud binary data
    sensor_msgs::PointCloud2::Ptr msg_ptr(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(point_cloud, *msg_ptr);
    pub_.publish(msg_ptr);
  }
};

template<>
class Publisher<sensor_msgs::PointCloud2>: public BasePublisher
{
public:
  Publisher() {}

  Publisher(ros::NodeHandle & nh, const std::string & topic, uint32_t queue_size)
  {
    advertise(nh, topic, queue_size);
  }

  ~Publisher() {}

  void
  publish(const sensor_msgs::PointCloud2Ptr & point_cloud) const
  {
    pub_.publish(point_cloud);
    // pub_.publish (*point_cloud);
  }

  void
  publish(const sensor_msgs::PointCloud2 & point_cloud) const
  {
    pub_.publish(boost::make_shared<const sensor_msgs::PointCloud2>(point_cloud));
    // pub_.publish (point_cloud);
  }
};
}  // namespace pcl_ros
#endif  // PCL_ROS__PUBLISHER_HPP_
