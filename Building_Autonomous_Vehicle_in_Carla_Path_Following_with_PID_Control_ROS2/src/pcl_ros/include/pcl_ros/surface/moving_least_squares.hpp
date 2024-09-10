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
 * $Id: moving_least_squares.h 36097 2011-02-20 14:18:58Z marton $
 *
 */

#ifndef PCL_ROS__SURFACE__MOVING_LEAST_SQUARES_HPP_
#define PCL_ROS__SURFACE__MOVING_LEAST_SQUARES_HPP_

#include <pcl/surface/mls.h>
#include <dynamic_reconfigure/server.h>
#include "pcl_ros/pcl_nodelet.hpp"
#include "pcl_ros/MLSConfig.hpp"

namespace pcl_ros
{
namespace sync_policies = message_filters::sync_policies;

/** \brief @b MovingLeastSquares represents a nodelet using the MovingLeastSquares implementation.
  * The type of the output is the same as the input, it only smooths the XYZ coordinates according
  * to the parameters.
  * Normals are estimated at each point as well and published on a separate topic.
  * \author Radu Bogdan Rusu, Zoltan-Csaba Marton
  */
class MovingLeastSquares : public PCLNodelet
{
  typedef pcl::PointXYZ PointIn;
  typedef pcl::PointNormal NormalOut;

  typedef pcl::PointCloud<PointIn> PointCloudIn;
  typedef boost::shared_ptr<PointCloudIn> PointCloudInPtr;
  typedef boost::shared_ptr<const PointCloudIn> PointCloudInConstPtr;
  typedef pcl::PointCloud<NormalOut> NormalCloudOut;

  typedef pcl::KdTree<PointIn> KdTree;
  typedef pcl::KdTree<PointIn>::Ptr KdTreePtr;

protected:
  /** \brief An input point cloud describing the surface that is to be used for
    * nearest neighbors estimation.
    */
  PointCloudInConstPtr surface_;

  /** \brief A pointer to the spatial search object. */
  KdTreePtr tree_;

  /** \brief The nearest neighbors search radius for each point. */
  double search_radius_;

  /** \brief The number of K nearest neighbors to use for each point. */
  // int k_;

  /** \brief Whether to use a polynomial fit. */
  bool use_polynomial_fit_;

  /** \brief The order of the polynomial to be fit. */
  int polynomial_order_;

  /** \brief How 'flat' should the neighbor weighting gaussian be
    * the smaller, the more local the fit).
    */
  double gaussian_parameter_;

  // ROS nodelet attributes
  /** \brief The surface PointCloud subscriber filter. */
  message_filters::Subscriber<PointCloudIn> sub_surface_filter_;

  /** \brief Parameter for the spatial locator tree. By convention, the values represent:
    * 0: ANN (Approximate Nearest Neigbor library) kd-tree
    * 1: FLANN (Fast Library for Approximate Nearest Neighbors) kd-tree
    * 2: Organized spatial dataset index
    */
  int spatial_locator_type_;

  /** \brief Pointer to a dynamic reconfigure service. */
  boost::shared_ptr<dynamic_reconfigure::Server<MLSConfig>> srv_;

  /** \brief Dynamic reconfigure callback
    * \param config the config object
    * \param level the dynamic reconfigure level
    */
  void config_callback(MLSConfig & config, uint32_t level);

  /** \brief Nodelet initialization routine. */
  virtual void onInit();

  /** \brief LazyNodelet connection routine. */
  virtual void subscribe();
  virtual void unsubscribe();

private:
  /** \brief Input point cloud callback.
    * \param cloud the pointer to the input point cloud
    * \param indices the pointer to the input point cloud indices
    */
  void input_indices_callback(
    const PointCloudInConstPtr & cloud,
    const PointIndicesConstPtr & indices);

private:
  /** \brief The PCL implementation used. */
  pcl::MovingLeastSquares<PointIn, NormalOut> impl_;

  /** \brief The input PointCloud subscriber. */
  ros::Subscriber sub_input_;

  /** \brief The output PointCloud (containing the normals) publisher. */
  ros::Publisher pub_normals_;

  /** \brief Synchronized input, and indices.*/
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ExactTime<PointCloudIn,
    PointIndices>>> sync_input_indices_e_;
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ApproximateTime<PointCloudIn,
    PointIndices>>> sync_input_indices_a_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace pcl_ros

#endif  // PCL_ROS__SURFACE__MOVING_LEAST_SQUARES_HPP_
