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
 * $Id: fpfh_omp.h 35361 2011-01-20 04:34:49Z rusu $
 *
 */

#ifndef PCL_ROS__FEATURES__FPFH_OMP_HPP_
#define PCL_ROS__FEATURES__FPFH_OMP_HPP_

#include <pcl/features/fpfh_omp.h>
#include "pcl_ros/features/fpfh.hpp"

namespace pcl_ros
{
/** \brief @b FPFHEstimationOMP estimates the Fast Point Feature Histogram (FPFH) descriptor for a given point cloud
  * dataset containing points and normals, in parallel, using the OpenMP standard.
  *
  * @note If you use this code in any academic work, please cite:
  *
  * <ul>
  * <li> R.B. Rusu, N. Blodow, M. Beetz.
  *      Fast Point Feature Histograms (FPFH) for 3D Registration.
  *      In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA),
  *      Kobe, Japan, May 12-17 2009.
  * </li>
  * <li> R.B. Rusu, A. Holzbach, N. Blodow, M. Beetz.
  *      Fast Geometric Point Labeling using Conditional Random Fields.
  *      In Proceedings of the 22nd IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),
  *      St. Louis, MO, USA, October 11-15 2009.
  * </li>
  * </ul>
  * \author Radu Bogdan Rusu
  */
class FPFHEstimationOMP : public FeatureFromNormals
{
private:
  pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> impl_;

  typedef pcl::PointCloud<pcl::FPFHSignature33> PointCloudOut;

  /** \brief Child initialization routine. Internal method. */
  inline bool
  childInit(ros::NodeHandle & nh)
  {
    // Create the output publisher
    pub_output_ = advertise<PointCloudOut>(nh, "output", max_queue_size_);
    return true;
  }

  /** \brief Publish an empty point cloud of the feature output type. */
  void emptyPublish(const PointCloudInConstPtr & cloud);

  /** \brief Compute the feature and publish it. */
  void computePublish(
    const PointCloudInConstPtr & cloud,
    const PointCloudNConstPtr & normals,
    const PointCloudInConstPtr & surface,
    const IndicesPtr & indices);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace pcl_ros

#endif  // PCL_ROS__FEATURES__FPFH_OMP_HPP_
