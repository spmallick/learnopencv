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
 * $Id: concatenate_data.h 35231 2011-01-14 05:33:20Z rusu $
 *
 */

#ifndef PCL_ROS__IO__CONCATENATE_DATA_HPP_
#define PCL_ROS__IO__CONCATENATE_DATA_HPP_

// ROS includes
#include <tf/transform_listener.h>
#include <nodelet_topic_tools/nodelet_lazy.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/pass_through.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <string>
#include <vector>

namespace pcl_ros
{
namespace sync_policies = message_filters::sync_policies;

/** \brief @b PointCloudConcatenateFieldsSynchronizer is a special form of data
  * synchronizer: it listens for a set of input PointCloud messages on the same topic,
  * checks their timestamps, and concatenates their fields together into a single
  * PointCloud output message.
  * \author Radu Bogdan Rusu
  */
class PointCloudConcatenateDataSynchronizer : public nodelet_topic_tools::NodeletLazy
{
public:
  typedef sensor_msgs::PointCloud2 PointCloud2;
  typedef PointCloud2::Ptr PointCloud2Ptr;
  typedef PointCloud2::ConstPtr PointCloud2ConstPtr;

  /** \brief Empty constructor. */
  PointCloudConcatenateDataSynchronizer()
  : maximum_queue_size_(3) {}

  /** \brief Empty constructor.
    * \param queue_size the maximum queue size
    */
  explicit PointCloudConcatenateDataSynchronizer(int queue_size)
  : maximum_queue_size_(queue_size), approximate_sync_(false) {}

  /** \brief Empty destructor. */
  virtual ~PointCloudConcatenateDataSynchronizer() {}

  void onInit();
  void subscribe();
  void unsubscribe();

private:
  /** \brief The output PointCloud publisher. */
  ros::Publisher pub_output_;

  /** \brief The maximum number of messages that we can store in the queue. */
  int maximum_queue_size_;

  /** \brief True if we use an approximate time synchronizer
    * versus an exact one (false by default).
    */
  bool approximate_sync_;

  /** \brief A vector of message filters. */
  std::vector<boost::shared_ptr<message_filters::Subscriber<PointCloud2>>> filters_;

  /** \brief Output TF frame the concatenated points should be transformed to. */
  std::string output_frame_;

  /** \brief Input point cloud topics. */
  XmlRpc::XmlRpcValue input_topics_;

  /** \brief TF listener object. */
  tf::TransformListener tf_;

  /** \brief Null passthrough filter, used for pushing empty elements in the
    * synchronizer */
  message_filters::PassThrough<PointCloud2> nf_;

  /** \brief Synchronizer.
    * \note This will most likely be rewritten soon using the DynamicTimeSynchronizer.
    */
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ApproximateTime<PointCloud2,
    PointCloud2, PointCloud2, PointCloud2, PointCloud2, PointCloud2, PointCloud2,
    PointCloud2>>> ts_a_;
  boost::shared_ptr<message_filters::Synchronizer<sync_policies::ExactTime<PointCloud2, PointCloud2,
    PointCloud2, PointCloud2, PointCloud2, PointCloud2, PointCloud2, PointCloud2>>> ts_e_;

  /** \brief Input point cloud callback.
    * Because we want to use the same synchronizer object, we push back
    * empty elements with the same timestamp.
    */
  inline void
  input_callback(const PointCloud2ConstPtr & input)
  {
    PointCloud2 cloud;
    cloud.header.stamp = input->header.stamp;
    nf_.add(boost::make_shared<PointCloud2>(cloud));
  }

  /** \brief Input callback for 8 synchronized topics. */
  void input(
    const PointCloud2::ConstPtr & in1, const PointCloud2::ConstPtr & in2,
    const PointCloud2::ConstPtr & in3, const PointCloud2::ConstPtr & in4,
    const PointCloud2::ConstPtr & in5, const PointCloud2::ConstPtr & in6,
    const PointCloud2::ConstPtr & in7, const PointCloud2::ConstPtr & in8);

  void combineClouds(const PointCloud2 & in1, const PointCloud2 & in2, PointCloud2 & out);
};
}  // namespace pcl_ros

#endif  // PCL_ROS__IO__CONCATENATE_DATA_HPP_
