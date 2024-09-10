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
 * $Id: io.cpp 35361 2011-01-20 04:34:49Z rusu $
 *
 */

#include <pluginlib/class_list_macros.h>
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
// #include <pcl_ros/subscriber.hpp>
#include <nodelet_topic_tools/nodelet_mux.h>
#include <nodelet_topic_tools/nodelet_demux.h>

typedef nodelet::NodeletMUX<sensor_msgs::PointCloud2,
    message_filters::Subscriber<sensor_msgs::PointCloud2>> NodeletMUX;
// typedef nodelet::NodeletDEMUX<sensor_msgs::PointCloud2,
//    pcl_ros::Subscriber<sensor_msgs::PointCloud2> > NodeletDEMUX;
typedef nodelet::NodeletDEMUX<sensor_msgs::PointCloud2> NodeletDEMUX;

// #include "pcd_io.cpp"
// #include "bag_io.cpp"
// #include "concatenate_fields.cpp"
// #include "concatenate_data.cpp"

PLUGINLIB_EXPORT_CLASS(NodeletMUX, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(NodeletDEMUX, nodelet::Nodelet);
// PLUGINLIB_EXPORT_CLASS(NodeletDEMUX_ROS,nodelet::Nodelet);
