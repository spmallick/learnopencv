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
 * $Id: surface_convex_hull.cpp 34612 2010-12-08 01:06:27Z rusu $
 *
 */

/**
 \author Ethan Rublee
 **/
// ROS core
#include <ros/ros.h>
// Image message
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
// pcl::toROSMsg
#include <pcl/io/pcd_io.h>
// conversions from PCL custom types
#include <pcl_conversions/pcl_conversions.h>
// stl stuff
#include <string>

class PointCloudToImage
{
public:
  void
  cloud_cb(const sensor_msgs::PointCloud2ConstPtr & cloud)
  {
    if (cloud->height <= 1) {
      ROS_ERROR("Input point cloud is not organized, ignoring!");
      return;
    }
    try {
      pcl::toROSMsg(*cloud, image_);  // convert the cloud
      image_.header = cloud->header;
      image_pub_.publish(image_);  // publish our cloud image
    } catch (std::runtime_error & e) {
      ROS_ERROR_STREAM(
        "Error in converting cloud to image message: " <<
          e.what());
    }
  }
  PointCloudToImage()
  : cloud_topic_("input"), image_topic_("output")
  {
    sub_ = nh_.subscribe(
      cloud_topic_, 30,
      &PointCloudToImage::cloud_cb, this);
    image_pub_ = nh_.advertise<sensor_msgs::Image>(image_topic_, 30);

    // print some info about the node
    std::string r_ct = nh_.resolveName(cloud_topic_);
    std::string r_it = nh_.resolveName(image_topic_);
    ROS_INFO_STREAM("Listening for incoming data on topic " << r_ct);
    ROS_INFO_STREAM("Publishing image on topic " << r_it);
  }

private:
  ros::NodeHandle nh_;
  sensor_msgs::Image image_;  // cache the image message
  std::string cloud_topic_;  // default input
  std::string image_topic_;  // default output
  ros::Subscriber sub_;  // cloud subscriber
  ros::Publisher image_pub_;  // image message publisher
};

int
main(int argc, char ** argv)
{
  ros::init(argc, argv, "convert_pointcloud_to_image");
  PointCloudToImage pci;  // this loads up the node
  ros::spin();  // where she stops nobody knows
  return 0;
}
