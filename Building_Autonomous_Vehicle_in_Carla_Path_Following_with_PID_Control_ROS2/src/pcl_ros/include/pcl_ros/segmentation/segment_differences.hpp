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
 * $Id: segment_differences.h 35361 2011-01-20 04:34:49Z rusu $
 *
 */

#ifndef PCL_ROS__SEGMENTATION__SEGMENT_DIFFERENCES_HPP_
#define PCL_ROS__SEGMENTATION__SEGMENT_DIFFERENCES_HPP_

#include <pcl/segmentation/segment_differences.h>
#include <dynamic_reconfigure/server.h>
#include "pcl_ros/SegmentDifferencesConfig.hpp"
#include "pcl_ros/pcl_nodelet.hpp"


namespace pcl_ros
{
namespace sync_policies = message_filters::sync_policies;

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b SegmentDifferences obtains the difference between two spatially aligned point clouds and returns the
  * difference between them for a maximum given distance threshold.
  * \author Radu Bogdan Rusu
  */
class SegmentDifferences : public PCLNodelet
{
  typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
  typedef boost::shared_ptr<PointCloud> PointCloudPtr;
  typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

public:
  /** \brief Empty constructor. */
  SegmentDifferences() {}

protected:
  /** \brief The message filter subscriber for PointCloud2. */
  message_filters::Subscriber<PointCloud> sub_target_filter_;

  /** \brief Synchronized input, and planar hull.*/
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ExactTime<PointCloud,
    PointCloud>>> sync_input_target_e_;
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ApproximateTime<PointCloud,
    PointCloud>>> sync_input_target_a_;

  /** \brief Pointer to a dynamic reconfigure service. */
  boost::shared_ptr<dynamic_reconfigure::Server<SegmentDifferencesConfig>> srv_;

  /** \brief Nodelet initialization routine. */
  void onInit();

  /** \brief LazyNodelet connection routine. */
  void subscribe();
  void unsubscribe();

  /** \brief Dynamic reconfigure callback
    * \param config the config object
    * \param level the dynamic reconfigure level
    */
  void config_callback(SegmentDifferencesConfig & config, uint32_t level);

  /** \brief Input point cloud callback.
    * \param cloud the pointer to the input point cloud
    * \param cloud_target the pointcloud that we want to segment \a cloud from
    */
  void input_target_callback(
    const PointCloudConstPtr & cloud,
    const PointCloudConstPtr & cloud_target);

private:
  /** \brief The PCL implementation used. */
  pcl::SegmentDifferences<pcl::PointXYZ> impl_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace pcl_ros

#endif  // PCL_ROS__SEGMENTATION__SEGMENT_DIFFERENCES_HPP_
