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
 * $Id: sac_segmentation.h 35564 2011-01-27 07:32:12Z rusu $
 *
 */

#ifndef PCL_ROS__SEGMENTATION__SAC_SEGMENTATION_HPP_
#define PCL_ROS__SEGMENTATION__SAC_SEGMENTATION_HPP_

#include <message_filters/pass_through.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <dynamic_reconfigure/server.h>
#include <string>
#include "pcl_ros/pcl_nodelet.hpp"
#include "pcl_ros/SACSegmentationConfig.hpp"
#include "pcl_ros/SACSegmentationFromNormalsConfig.hpp"

namespace pcl_ros
{
namespace sync_policies = message_filters::sync_policies;

////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b SACSegmentation represents the Nodelet segmentation class for Sample Consensus
  * methods and models, in the sense that it just creates a Nodelet wrapper for generic-purpose
  * SAC-based segmentation.
  * \author Radu Bogdan Rusu
  */
class SACSegmentation : public PCLNodelet
{
  typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
  typedef boost::shared_ptr<PointCloud> PointCloudPtr;
  typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

public:
  /** \brief Constructor. */
  SACSegmentation()
  : min_inliers_(0) {}

  /** \brief Set the input TF frame the data should be transformed into before processing,
    * if input.header.frame_id is different.
    * \param tf_frame the TF frame the input PointCloud should be transformed into before processing
    */
  inline void setInputTFframe(std::string tf_frame) {tf_input_frame_ = tf_frame;}

  /** \brief Get the TF frame the input PointCloud should be transformed into before processing. */
  inline std::string getInputTFframe() {return tf_input_frame_;}

  /** \brief Set the output TF frame the data should be transformed into after processing.
    * \param tf_frame the TF frame the PointCloud should be transformed into after processing
    */
  inline void setOutputTFframe(std::string tf_frame) {tf_output_frame_ = tf_frame;}

  /** \brief Get the TF frame the PointCloud should be transformed into after processing. */
  inline std::string getOutputTFframe() {return tf_output_frame_;}

protected:
  // The minimum number of inliers a model must have in order to be considered valid.
  int min_inliers_;

  // ROS nodelet attributes
  /** \brief The output PointIndices publisher. */
  ros::Publisher pub_indices_;

  /** \brief The output ModelCoefficients publisher. */
  ros::Publisher pub_model_;

  /** \brief The input PointCloud subscriber. */
  ros::Subscriber sub_input_;

  /** \brief Pointer to a dynamic reconfigure service. */
  boost::shared_ptr<dynamic_reconfigure::Server<SACSegmentationConfig>> srv_;

  /** \brief The input TF frame the data should be transformed into,
    * if input.header.frame_id is different.
    */
  std::string tf_input_frame_;

  /** \brief The original data input TF frame. */
  std::string tf_input_orig_frame_;

  /** \brief The output TF frame the data should be transformed into,
    * if input.header.frame_id is different.
    */
  std::string tf_output_frame_;

  /** \brief Null passthrough filter, used for pushing empty elements in the
    * synchronizer */
  message_filters::PassThrough<pcl_msgs::PointIndices> nf_pi_;

  /** \brief Nodelet initialization routine. */
  virtual void onInit();

  /** \brief LazyNodelet connection routine. */
  virtual void subscribe();
  virtual void unsubscribe();

  /** \brief Dynamic reconfigure callback
    * \param config the config object
    * \param level the dynamic reconfigure level
    */
  void config_callback(SACSegmentationConfig & config, uint32_t level);

  /** \brief Input point cloud callback. Used when \a use_indices is set.
    * \param cloud the pointer to the input point cloud
    * \param indices the pointer to the input point cloud indices
    */
  void input_indices_callback(
    const PointCloudConstPtr & cloud,
    const PointIndicesConstPtr & indices);

  /** \brief Pointer to a set of indices stored internally.
   * (used when \a latched_indices_ is set).
   */
  PointIndices indices_;

  /** \brief Indices callback. Used when \a latched_indices_ is set.
    * \param indices the pointer to the input point cloud indices
    */
  inline void
  indices_callback(const PointIndicesConstPtr & indices)
  {
    indices_ = *indices;
  }

  /** \brief Input callback. Used when \a latched_indices_ is set.
    * \param input the pointer to the input point cloud
    */
  inline void
  input_callback(const PointCloudConstPtr & input)
  {
    indices_.header = fromPCL(input->header);
    PointIndicesConstPtr indices;
    indices.reset(new PointIndices(indices_));
    nf_pi_.add(indices);
  }

private:
  /** \brief Internal mutex. */
  boost::mutex mutex_;

  /** \brief The PCL implementation used. */
  pcl::SACSegmentation<pcl::PointXYZ> impl_;

  /** \brief Synchronized input, and indices.*/
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ExactTime<PointCloud,
    PointIndices>>> sync_input_indices_e_;
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ApproximateTime<PointCloud,
    PointIndices>>> sync_input_indices_a_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b SACSegmentationFromNormals represents the PCL nodelet segmentation class for
  * Sample Consensus methods and models that require the use of surface normals for estimation.
  */
class SACSegmentationFromNormals : public SACSegmentation
{
  typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
  typedef boost::shared_ptr<PointCloud> PointCloudPtr;
  typedef boost::shared_ptr<const PointCloud> PointCloudConstPtr;

  typedef pcl::PointCloud<pcl::Normal> PointCloudN;
  typedef boost::shared_ptr<PointCloudN> PointCloudNPtr;
  typedef boost::shared_ptr<const PointCloudN> PointCloudNConstPtr;

public:
  /** \brief Set the input TF frame the data should be transformed into before processing,
    * if input.header.frame_id is different.
    * \param tf_frame the TF frame the input PointCloud should be transformed into before processing
    */
  inline void setInputTFframe(std::string tf_frame) {tf_input_frame_ = tf_frame;}

  /** \brief Get the TF frame the input PointCloud should be transformed into before processing. */
  inline std::string getInputTFframe() {return tf_input_frame_;}

  /** \brief Set the output TF frame the data should be transformed into after processing.
    * \param tf_frame the TF frame the PointCloud should be transformed into after processing
    */
  inline void setOutputTFframe(std::string tf_frame) {tf_output_frame_ = tf_frame;}

  /** \brief Get the TF frame the PointCloud should be transformed into after processing. */
  inline std::string getOutputTFframe() {return tf_output_frame_;}

protected:
  // ROS nodelet attributes
  /** \brief The normals PointCloud subscriber filter. */
  message_filters::Subscriber<PointCloudN> sub_normals_filter_;

  /** \brief The input PointCloud subscriber. */
  ros::Subscriber sub_axis_;

  /** \brief Pointer to a dynamic reconfigure service. */
  boost::shared_ptr<dynamic_reconfigure::Server<SACSegmentationFromNormalsConfig>> srv_;

  /** \brief Input point cloud callback.
    * Because we want to use the same synchronizer object, we push back
    * empty elements with the same timestamp.
    */
  inline void
  input_callback(const PointCloudConstPtr & cloud)
  {
    PointIndices indices;
    indices.header.stamp = fromPCL(cloud->header).stamp;
    nf_.add(boost::make_shared<PointIndices>(indices));
  }

  /** \brief Null passthrough filter, used for pushing empty elements in the
    * synchronizer */
  message_filters::PassThrough<PointIndices> nf_;

  /** \brief The input TF frame the data should be transformed into,
   * if input.header.frame_id is different.
   */
  std::string tf_input_frame_;
  /** \brief The original data input TF frame. */
  std::string tf_input_orig_frame_;
  /** \brief The output TF frame the data should be transformed into,
    * if input.header.frame_id is different.
    */
  std::string tf_output_frame_;

  /** \brief Nodelet initialization routine. */
  virtual void onInit();

  /** \brief LazyNodelet connection routine. */
  virtual void subscribe();
  virtual void unsubscribe();

  /** \brief Model callback
    * \param model the sample consensus model found
    */
  void axis_callback(const pcl_msgs::ModelCoefficientsConstPtr & model);

  /** \brief Dynamic reconfigure callback
    * \param config the config object
    * \param level the dynamic reconfigure level
    */
  void config_callback(SACSegmentationFromNormalsConfig & config, uint32_t level);

  /** \brief Input point cloud callback.
    * \param cloud the pointer to the input point cloud
    * \param cloud_normals the pointer to the input point cloud normals
    * \param indices the pointer to the input point cloud indices
    */
  void input_normals_indices_callback(
    const PointCloudConstPtr & cloud,
    const PointCloudNConstPtr & cloud_normals,
    const PointIndicesConstPtr & indices);

private:
  /** \brief Internal mutex. */
  boost::mutex mutex_;

  /** \brief The PCL implementation used. */
  pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> impl_;

  /** \brief Synchronized input, normals, and indices.*/
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ApproximateTime<PointCloud,
    PointCloudN, PointIndices>>> sync_input_normals_indices_a_;
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ExactTime<PointCloud, PointCloudN,
    PointIndices>>> sync_input_normals_indices_e_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace pcl_ros

#endif  // PCL_ROS__SEGMENTATION__SAC_SEGMENTATION_HPP_
