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
 * $Id: extract_clusters.h 35361 2011-01-20 04:34:49Z rusu $
 *
 */

#ifndef PCL_ROS__SEGMENTATION__EXTRACT_CLUSTERS_HPP_
#define PCL_ROS__SEGMENTATION__EXTRACT_CLUSTERS_HPP_

#include <dynamic_reconfigure/server.h>
#include <pcl/segmentation/extract_clusters.h>
#include <limits>
#include "pcl_ros/pcl_nodelet.hpp"
#include "pcl_ros/EuclideanClusterExtractionConfig.hpp"

namespace pcl_ros
{
namespace sync_policies = message_filters::sync_policies;

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b EuclideanClusterExtraction represents a segmentation class for cluster extraction in an Euclidean sense.
  * \author Radu Bogdan Rusu
  */
class EuclideanClusterExtraction : public PCLNodelet
{
public:
  /** \brief Empty constructor. */
  EuclideanClusterExtraction()
  : publish_indices_(false), max_clusters_(std::numeric_limits<int>::max()) {}

protected:
  // ROS nodelet attributes
  /** \brief Publish indices or convert to PointCloud clusters. Default: false */
  bool publish_indices_;

  /** \brief Maximum number of clusters to publish. */
  int max_clusters_;

  /** \brief Pointer to a dynamic reconfigure service. */
  boost::shared_ptr<dynamic_reconfigure::Server<EuclideanClusterExtractionConfig>> srv_;

  /** \brief Nodelet initialization routine. */
  void onInit();

  /** \brief LazyNodelet connection routine. */
  void subscribe();
  void unsubscribe();

  /** \brief Dynamic reconfigure callback
    * \param config the config object
    * \param level the dynamic reconfigure level
    */
  void config_callback(EuclideanClusterExtractionConfig & config, uint32_t level);

  /** \brief Input point cloud callback.
    * \param cloud the pointer to the input point cloud
    * \param indices the pointer to the input point cloud indices
    */
  void input_indices_callback(
    const PointCloudConstPtr & cloud,
    const PointIndicesConstPtr & indices);

private:
  /** \brief The PCL implementation used. */
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> impl_;

  /** \brief The input PointCloud subscriber. */
  ros::Subscriber sub_input_;

  /** \brief Synchronized input, and indices.*/
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ExactTime<PointCloud,
    PointIndices>>> sync_input_indices_e_;
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ApproximateTime<PointCloud,
    PointIndices>>> sync_input_indices_a_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace pcl_ros

#endif  // PCL_ROS__SEGMENTATION__EXTRACT_CLUSTERS_HPP_
