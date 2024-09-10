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
 * $Id: bag_io.h 35471 2011-01-25 06:50:00Z rusu $
 *
 */

#ifndef PCL_ROS__IO__BAG_IO_HPP_
#define PCL_ROS__IO__BAG_IO_HPP_

#include <sensor_msgs/PointCloud2.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <string>
#include <pcl_ros/pcl_nodelet.hpp>

namespace pcl_ros
{
////////////////////////////////////////////////////////////////////////////////////////////
/** \brief BAG PointCloud file format reader.
  * \author Radu Bogdan Rusu
  */
class BAGReader : public nodelet::Nodelet
{
public:
  typedef sensor_msgs::PointCloud2 PointCloud;
  typedef PointCloud::Ptr PointCloudPtr;
  typedef PointCloud::ConstPtr PointCloudConstPtr;

  /** \brief Empty constructor. */
  BAGReader()
  : publish_rate_(0), output_() /*, cloud_received_ (false)*/ {}

  /** \brief Set the publishing rate in seconds.
    * \param publish_rate the publishing rate in seconds
    */
  inline void setPublishRate(double publish_rate) {publish_rate_ = publish_rate;}

  /** \brief Get the publishing rate in seconds. */
  inline double getPublishRate() {return publish_rate_;}

  /** \brief Get the next point cloud dataset in the BAG file.
    * \return the next point cloud dataset as read from the file
    */
  inline PointCloudConstPtr
  getNextCloud()
  {
    if (it_ != view_.end()) {
      output_ = it_->instantiate<sensor_msgs::PointCloud2>();
      ++it_;
    }
    return output_;
  }

  /** \brief Open a BAG file for reading and select a specified topic
    * \param file_name the BAG file to open
    * \param topic_name the topic that we want to read data from
    */
  bool open(const std::string & file_name, const std::string & topic_name);

  /** \brief Close an open BAG file. */
  inline void
  close()
  {
    bag_.close();
  }

  /** \brief Nodelet initialization routine. */
  virtual void onInit();

private:
  /** \brief The publishing interval in seconds. Set to 0 to publish once (default). */
  double publish_rate_;

  /** \brief The BAG object. */
  rosbag::Bag bag_;

  /** \brief The BAG view object. */
  rosbag::View view_;

  /** \brief The BAG view iterator object. */
  rosbag::View::iterator it_;

  /** \brief The name of the topic that contains the PointCloud data. */
  std::string topic_name_;

  /** \brief The name of the BAG file that contains the PointCloud data. */
  std::string file_name_;

  /** \brief The output point cloud dataset containing the points loaded from the file. */
  PointCloudPtr output_;

  /** \brief Signals that a new PointCloud2 message has been read from the file. */
  // bool cloud_received_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace pcl_ros

#endif  // PCL_ROS__IO__BAG_IO_HPP_
