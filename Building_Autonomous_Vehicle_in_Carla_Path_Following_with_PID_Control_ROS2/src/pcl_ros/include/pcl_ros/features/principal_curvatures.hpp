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
 * $Id: principal_curvatures.h 35361 2011-01-20 04:34:49Z rusu $
 *
 */

#ifndef PCL_ROS__FEATURES__PRINCIPAL_CURVATURES_HPP_
#define PCL_ROS__FEATURES__PRINCIPAL_CURVATURES_HPP_

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET true
#include <pcl/features/principal_curvatures.h>
#include "pcl_ros/features/feature.hpp"

namespace pcl_ros
{
/** \brief @b PrincipalCurvaturesEstimation estimates the directions (eigenvectors) and magnitudes (eigenvalues) of
  * principal surface curvatures for a given point cloud dataset containing points and normals.
  *
  * @note The code is stateful as we do not expect this class to be multicore parallelized. Please look at
  * \a NormalEstimationOpenMP and \a NormalEstimationTBB for examples on how to extend this to parallel implementations.
  * \author Radu Bogdan Rusu, Jared Glover
  */
class PrincipalCurvaturesEstimation : public FeatureFromNormals
{
private:
  pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> impl_;

  typedef pcl::PointCloud<pcl::PrincipalCurvatures> PointCloudOut;

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

#endif  // PCL_ROS__FEATURES__PRINCIPAL_CURVATURES_HPP_
