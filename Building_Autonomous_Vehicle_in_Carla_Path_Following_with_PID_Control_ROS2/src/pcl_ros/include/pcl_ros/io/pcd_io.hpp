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
 * $Id: pcd_io.h 35054 2011-01-03 21:16:49Z rusu $
 *
 */

#ifndef PCL_ROS__IO__PCD_IO_HPP_
#define PCL_ROS__IO__PCD_IO_HPP_

#include <pcl/io/pcd_io.h>
#include <string>
#include "pcl_ros/pcl_nodelet.hpp"

namespace pcl_ros
{
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/** \brief Point Cloud Data (PCD) file format reader.
  * \author Radu Bogdan Rusu
  */
class PCDReader : public PCLNodelet
{
public:
  typedef sensor_msgs::PointCloud2 PointCloud2;
  typedef PointCloud2::Ptr PointCloud2Ptr;
  typedef PointCloud2::ConstPtr PointCloud2ConstPtr;

  /** \brief Empty constructor. */
  PCDReader()
  : publish_rate_(0), tf_frame_("/base_link") {}

  virtual void onInit();

  /** \brief Set the publishing rate in seconds.
    * \param publish_rate the publishing rate in seconds
    */
  inline void setPublishRate(double publish_rate) {publish_rate_ = publish_rate;}

  /** \brief Get the publishing rate in seconds. */
  inline double getPublishRate() {return publish_rate_;}

  /** \brief Set the TF frame the PointCloud will be published in.
    * \param tf_frame the TF frame the PointCloud will be published in
    */
  inline void setTFframe(std::string tf_frame) {tf_frame_ = tf_frame;}

  /** \brief Get the TF frame the PointCloud is published in. */
  inline std::string getTFframe() {return tf_frame_;}

protected:
  /** \brief The publishing interval in seconds. Set to 0 to only publish once (default). */
  double publish_rate_;

  /** \brief The TF frame the data should be published in ("/base_link" by default). */
  std::string tf_frame_;

  /** \brief The name of the file that contains the PointCloud data. */
  std::string file_name_;

  /** \brief The output point cloud dataset containing the points loaded from the file. */
  PointCloud2Ptr output_;

private:
  /** \brief The PCL implementation used. */
  pcl::PCDReader impl_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/** \brief Point Cloud Data (PCD) file format writer.
  * \author Radu Bogdan Rusu
  */
class PCDWriter : public PCLNodelet
{
public:
  PCDWriter()
  : file_name_(""), binary_mode_(true) {}

  typedef sensor_msgs::PointCloud2 PointCloud2;
  typedef PointCloud2::Ptr PointCloud2Ptr;
  typedef PointCloud2::ConstPtr PointCloud2ConstPtr;

  virtual void onInit();
  void input_callback(const PointCloud2ConstPtr & cloud);

  /** \brief The input PointCloud subscriber. */
  ros::Subscriber sub_input_;

protected:
  /** \brief The name of the file that contains the PointCloud data. */
  std::string file_name_;

  /** \brief Set to true if the output files should be saved in binary mode (true). */
  bool binary_mode_;

private:
  /** \brief The PCL implementation used. */
  pcl::PCDWriter impl_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace pcl_ros

#endif  // PCL_ROS__IO__PCD_IO_HPP_
