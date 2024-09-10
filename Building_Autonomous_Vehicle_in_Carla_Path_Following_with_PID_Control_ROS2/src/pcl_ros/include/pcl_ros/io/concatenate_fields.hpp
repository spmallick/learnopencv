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
 * $Id: concatenate_fields.h 35052 2011-01-03 21:04:57Z rusu $
 *
 */

#ifndef PCL_ROS__IO__CONCATENATE_FIELDS_HPP_
#define PCL_ROS__IO__CONCATENATE_FIELDS_HPP_

// ROS includes
#include <nodelet_topic_tools/nodelet_lazy.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/PointCloud2.h>

#include <map>
#include <vector>

namespace pcl_ros
{
/** \brief @b PointCloudConcatenateFieldsSynchronizer is a special form of data synchronizer: it listens for a set of
  * input PointCloud messages on the same topic, checks their timestamps, and concatenates their fields together into
  * a single PointCloud output message.
  * \author Radu Bogdan Rusu
  */
class PointCloudConcatenateFieldsSynchronizer : public nodelet_topic_tools::NodeletLazy
{
public:
  typedef sensor_msgs::PointCloud2 PointCloud;
  typedef PointCloud::Ptr PointCloudPtr;
  typedef PointCloud::ConstPtr PointCloudConstPtr;

  /** \brief Empty constructor. */
  PointCloudConcatenateFieldsSynchronizer()
  : maximum_queue_size_(3), maximum_seconds_(0) {}

  /** \brief Empty constructor.
    * \param queue_size the maximum queue size
    */
  explicit PointCloudConcatenateFieldsSynchronizer(int queue_size)
  : maximum_queue_size_(queue_size), maximum_seconds_(0) {}

  /** \brief Empty destructor. */
  virtual ~PointCloudConcatenateFieldsSynchronizer() {}

  void onInit();
  void subscribe();
  void unsubscribe();
  void input_callback(const PointCloudConstPtr & cloud);

private:
  /** \brief The input PointCloud subscriber. */
  ros::Subscriber sub_input_;

  /** \brief The output PointCloud publisher. */
  ros::Publisher pub_output_;

  /** \brief The number of input messages that we expect on the input topic. */
  int input_messages_;

  /** \brief The maximum number of messages that we can store in the queue. */
  int maximum_queue_size_;

  /** \brief The maximum number of seconds to wait until we drop the synchronization. */
  double maximum_seconds_;

  /** \brief A queue for messages. */
  std::map<ros::Time, std::vector<PointCloudConstPtr>> queue_;
};
}  // namespace pcl_ros

#endif  // PCL_ROS__IO__CONCATENATE_FIELDS_HPP_
