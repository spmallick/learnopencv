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
 * $Id: feature.h 35422 2011-01-24 20:04:44Z rusu $
 *
 */

#ifndef PCL_ROS__FEATURES__FEATURE_HPP_
#define PCL_ROS__FEATURES__FEATURE_HPP_

// PCL includes
#include <pcl/features/feature.h>
#include <pcl_msgs/PointIndices.h>

#include <message_filters/pass_through.h>

// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>

// PCL conversions
#include <pcl_conversions/pcl_conversions.h>

#include "pcl_ros/pcl_nodelet.hpp"
#include "pcl_ros/FeatureConfig.hpp"

namespace pcl_ros
{
namespace sync_policies = message_filters::sync_policies;

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b Feature represents the base feature class. Some generic 3D operations that
  * are applicable to all features are defined here as static methods.
  * \author Radu Bogdan Rusu
  */
class Feature : public PCLNodelet
{
public:
  typedef pcl::KdTree<pcl::PointXYZ> KdTree;
  typedef pcl::KdTree<pcl::PointXYZ>::Ptr KdTreePtr;

  typedef pcl::PointCloud<pcl::PointXYZ> PointCloudIn;
  typedef boost::shared_ptr<PointCloudIn> PointCloudInPtr;
  typedef boost::shared_ptr<const PointCloudIn> PointCloudInConstPtr;

  typedef pcl::IndicesPtr IndicesPtr;
  typedef pcl::IndicesConstPtr IndicesConstPtr;

  /** \brief Empty constructor. */
  Feature()
  : /*input_(), indices_(), surface_(), */ tree_(), k_(0), search_radius_(0),
    use_surface_(false), spatial_locator_type_(-1)
  {}

protected:
  /** \brief The input point cloud dataset. */
  // PointCloudInConstPtr input_;

  /** \brief A pointer to the vector of point indices to use. */
  // IndicesConstPtr indices_;

  /** \brief An input point cloud describing the surface that is to be used
    * for nearest neighbors estimation.
    */
  // PointCloudInConstPtr surface_;

  /** \brief A pointer to the spatial search object. */
  KdTreePtr tree_;

  /** \brief The number of K nearest neighbors to use for each point. */
  int k_;

  /** \brief The nearest neighbors search radius for each point. */
  double search_radius_;

  // ROS nodelet attributes
  /** \brief The surface PointCloud subscriber filter. */
  message_filters::Subscriber<PointCloudIn> sub_surface_filter_;

  /** \brief The input PointCloud subscriber. */
  ros::Subscriber sub_input_;

  /** \brief Set to true if the nodelet needs to listen for incoming point
   * clouds representing the search surface.
   */
  bool use_surface_;

  /** \brief Parameter for the spatial locator tree. By convention, the values represent:
    * 0: ANN (Approximate Nearest Neigbor library) kd-tree
    * 1: FLANN (Fast Library for Approximate Nearest Neighbors) kd-tree
    * 2: Organized spatial dataset index
    */
  int spatial_locator_type_;

  /** \brief Pointer to a dynamic reconfigure service. */
  boost::shared_ptr<dynamic_reconfigure::Server<FeatureConfig>> srv_;

  /** \brief Child initialization routine. Internal method. */
  virtual bool childInit(ros::NodeHandle & nh) = 0;

  /** \brief Publish an empty point cloud of the feature output type. */
  virtual void emptyPublish(const PointCloudInConstPtr & cloud) = 0;

  /** \brief Compute the feature and publish it. Internal method. */
  virtual void computePublish(
    const PointCloudInConstPtr & cloud,
    const PointCloudInConstPtr & surface,
    const IndicesPtr & indices) = 0;

  /** \brief Dynamic reconfigure callback
    * \param config the config object
    * \param level the dynamic reconfigure level
    */
  void config_callback(FeatureConfig & config, uint32_t level);

  /** \brief Null passthrough filter, used for pushing empty elements in the
    * synchronizer */
  message_filters::PassThrough<PointIndices> nf_pi_;
  message_filters::PassThrough<PointCloudIn> nf_pc_;

  /** \brief Input point cloud callback.
    * Because we want to use the same synchronizer object, we push back
    * empty elements with the same timestamp.
    */
  inline void
  input_callback(const PointCloudInConstPtr & input)
  {
    PointIndices indices;
    indices.header.stamp = pcl_conversions::fromPCL(input->header).stamp;
    PointCloudIn cloud;
    cloud.header.stamp = input->header.stamp;
    nf_pc_.add(ros_ptr(cloud.makeShared()));
    nf_pi_.add(boost::make_shared<PointIndices>(indices));
  }

private:
  /** \brief Synchronized input, surface, and point indices.*/
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ApproximateTime<PointCloudIn,
    PointCloudIn, PointIndices>>> sync_input_surface_indices_a_;
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ExactTime<PointCloudIn,
    PointCloudIn, PointIndices>>> sync_input_surface_indices_e_;

  /** \brief Nodelet initialization routine. */
  virtual void onInit();

  /** \brief NodeletLazy connection routine. */
  virtual void subscribe();
  virtual void unsubscribe();

  /** \brief Input point cloud callback. Used when \a use_indices and \a use_surface are set.
    * \param cloud the pointer to the input point cloud
    * \param cloud_surface the pointer to the surface point cloud
    * \param indices the pointer to the input point cloud indices
    */
  void input_surface_indices_callback(
    const PointCloudInConstPtr & cloud,
    const PointCloudInConstPtr & cloud_surface,
    const PointIndicesConstPtr & indices);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
class FeatureFromNormals : public Feature
{
public:
  typedef sensor_msgs::PointCloud2 PointCloud2;

  typedef pcl::PointCloud<pcl::Normal> PointCloudN;
  typedef boost::shared_ptr<PointCloudN> PointCloudNPtr;
  typedef boost::shared_ptr<const PointCloudN> PointCloudNConstPtr;

  FeatureFromNormals()
  : normals_() {}

protected:
  /** \brief A pointer to the input dataset that contains the point normals of the XYZ dataset. */
  PointCloudNConstPtr normals_;

  /** \brief Child initialization routine. Internal method. */
  virtual bool childInit(ros::NodeHandle & nh) = 0;

  /** \brief Publish an empty point cloud of the feature output type. */
  virtual void emptyPublish(const PointCloudInConstPtr & cloud) = 0;

  /** \brief Compute the feature and publish it. */
  virtual void computePublish(
    const PointCloudInConstPtr & cloud,
    const PointCloudNConstPtr & normals,
    const PointCloudInConstPtr & surface,
    const IndicesPtr & indices) = 0;

private:
  /** \brief The normals PointCloud subscriber filter. */
  message_filters::Subscriber<PointCloudN> sub_normals_filter_;

  /** \brief Synchronized input, normals, surface, and point indices.*/
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ApproximateTime<PointCloudIn,
    PointCloudN, PointCloudIn, PointIndices>>> sync_input_normals_surface_indices_a_;
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ExactTime<PointCloudIn,
    PointCloudN, PointCloudIn, PointIndices>>> sync_input_normals_surface_indices_e_;

  /** \brief Internal method. */
  void computePublish(
    const PointCloudInConstPtr &,
    const PointCloudInConstPtr &,
    const IndicesPtr &) {}                        // This should never be called

  /** \brief Nodelet initialization routine. */
  virtual void onInit();

  /** \brief NodeletLazy connection routine. */
  virtual void subscribe();
  virtual void unsubscribe();

  /** \brief Input point cloud callback. Used when \a use_indices and \a use_surface are set.
    * \param cloud the pointer to the input point cloud
    * \param cloud_normals the pointer to the input point cloud normals
    * \param cloud_surface the pointer to the surface point cloud
    * \param indices the pointer to the input point cloud indices
    */
  void input_normals_surface_indices_callback(
    const PointCloudInConstPtr & cloud,
    const PointCloudNConstPtr & cloud_normals,
    const PointCloudInConstPtr & cloud_surface,
    const PointIndicesConstPtr & indices);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace pcl_ros

#endif  // PCL_ROS__FEATURES__FEATURE_HPP_
