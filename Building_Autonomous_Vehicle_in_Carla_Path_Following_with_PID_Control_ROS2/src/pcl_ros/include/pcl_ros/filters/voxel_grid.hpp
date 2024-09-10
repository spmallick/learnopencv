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
 * $Id: voxel_grid.h 35876 2011-02-09 01:04:36Z rusu $
 *
 */

#ifndef PCL_ROS__FILTERS__VOXEL_GRID_HPP_
#define PCL_ROS__FILTERS__VOXEL_GRID_HPP_

// PCL includes
#include <pcl/filters/voxel_grid.h>
#include <vector>
#include "pcl_ros/filters/filter.hpp"

namespace pcl_ros
{
/** \brief @b VoxelGrid assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
  * \author Radu Bogdan Rusu
  */
class VoxelGrid : public Filter
{
protected:
  /** \brief Call the actual filter.
    * \param input the input point cloud dataset
    * \param indices the input set of indices to use from \a input
    * \param output the resultant filtered dataset
    */
  inline void
  filter(
    const PointCloud2::ConstSharedPtr & input, const IndicesPtr & indices,
    PointCloud2 & output) override;

  /** \brief Parameter callback
    * \param params parameter values to set
    */
  rcl_interfaces::msg::SetParametersResult
  config_callback(const std::vector<rclcpp::Parameter> & params);

  OnSetParametersCallbackHandle::SharedPtr callback_handle_;

private:
  /** \brief The PCL filter implementation used. */
  pcl::VoxelGrid<pcl::PCLPointCloud2> impl_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit VoxelGrid(const rclcpp::NodeOptions & options);
};
}  // namespace pcl_ros

#endif  // PCL_ROS__FILTERS__VOXEL_GRID_HPP_
