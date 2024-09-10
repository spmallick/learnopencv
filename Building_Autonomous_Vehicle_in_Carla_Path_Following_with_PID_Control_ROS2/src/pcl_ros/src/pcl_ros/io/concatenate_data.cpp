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
 * $Id: concatenate_data.cpp 35231 2011-01-14 05:33:20Z rusu $
 *
 */

#include <pluginlib/class_list_macros.h>
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <string>
#include "pcl_ros/transforms.hpp"
#include "pcl_ros/io/concatenate_data.hpp"


//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PointCloudConcatenateDataSynchronizer::onInit()
{
  nodelet_topic_tools::NodeletLazy::onInit();

  // ---[ Mandatory parameters
  pnh_->getParam("output_frame", output_frame_);
  pnh_->getParam("approximate_sync", approximate_sync_);

  if (output_frame_.empty()) {
    NODELET_ERROR("[onInit] Need an 'output_frame' parameter to be set before continuing!");
    return;
  }

  if (!pnh_->getParam("input_topics", input_topics_)) {
    NODELET_ERROR("[onInit] Need a 'input_topics' parameter to be set before continuing!");
    return;
  }
  if (input_topics_.getType() != XmlRpc::XmlRpcValue::TypeArray) {
    NODELET_ERROR("[onInit] Invalid 'input_topics' parameter given!");
    return;
  }
  if (input_topics_.size() == 1) {
    NODELET_ERROR("[onInit] Only one topic given. Need at least two topics to continue.");
    return;
  }
  if (input_topics_.size() > 8) {
    NODELET_ERROR("[onInit] More than 8 topics passed!");
    return;
  }

  // ---[ Optional parameters
  pnh_->getParam("max_queue_size", maximum_queue_size_);

  // Output
  pub_output_ = advertise<PointCloud2>(*pnh_, "output", maximum_queue_size_);

  onInitPostProcess();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PointCloudConcatenateDataSynchronizer::subscribe()
{
  ROS_INFO_STREAM("Subscribing to " << input_topics_.size() << " user given topics as inputs:");
  for (int d = 0; d < input_topics_.size(); ++d) {
    ROS_INFO_STREAM(" - " << (std::string)(input_topics_[d]));
  }

  // Subscribe to the filters
  filters_.resize(input_topics_.size());

  // 8 topics
  if (approximate_sync_) {
    ts_a_.reset(
      new message_filters::Synchronizer<sync_policies::ApproximateTime<PointCloud2, PointCloud2,
      PointCloud2,
      PointCloud2, PointCloud2, PointCloud2,
      PointCloud2, PointCloud2>
      >(maximum_queue_size_));
  } else {
    ts_e_.reset(
      new message_filters::Synchronizer<sync_policies::ExactTime<PointCloud2, PointCloud2,
      PointCloud2,
      PointCloud2, PointCloud2, PointCloud2,
      PointCloud2, PointCloud2>
      >(maximum_queue_size_));
  }

  // First input_topics_.size () filters are valid
  for (int d = 0; d < input_topics_.size(); ++d) {
    filters_[d].reset(new message_filters::Subscriber<PointCloud2>());
    filters_[d]->subscribe(*pnh_, (std::string)(input_topics_[d]), maximum_queue_size_);
  }

  // Bogus null filter
  filters_[0]->registerCallback(
    bind(
      &PointCloudConcatenateDataSynchronizer::input_callback, this,
      _1));

  switch (input_topics_.size()) {
    case 2:
      {
        if (approximate_sync_) {
          ts_a_->connectInput(*filters_[0], *filters_[1], nf_, nf_, nf_, nf_, nf_, nf_);
        } else {
          ts_e_->connectInput(*filters_[0], *filters_[1], nf_, nf_, nf_, nf_, nf_, nf_);
        }
        break;
      }
    case 3:
      {
        if (approximate_sync_) {
          ts_a_->connectInput(*filters_[0], *filters_[1], *filters_[2], nf_, nf_, nf_, nf_, nf_);
        } else {
          ts_e_->connectInput(*filters_[0], *filters_[1], *filters_[2], nf_, nf_, nf_, nf_, nf_);
        }
        break;
      }
    case 4:
      {
        if (approximate_sync_) {
          ts_a_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], nf_, nf_, nf_,
            nf_);
        } else {
          ts_e_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], nf_, nf_, nf_,
            nf_);
        }
        break;
      }
    case 5:
      {
        if (approximate_sync_) {
          ts_a_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], *filters_[4],
            nf_, nf_, nf_);
        } else {
          ts_e_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], *filters_[4],
            nf_, nf_, nf_);
        }
        break;
      }
    case 6:
      {
        if (approximate_sync_) {
          ts_a_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], *filters_[4],
            *filters_[5], nf_, nf_);
        } else {
          ts_e_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], *filters_[4],
            *filters_[5], nf_, nf_);
        }
        break;
      }
    case 7:
      {
        if (approximate_sync_) {
          ts_a_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], *filters_[4],
            *filters_[5], *filters_[6], nf_);
        } else {
          ts_e_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], *filters_[4],
            *filters_[5], *filters_[6], nf_);
        }
        break;
      }
    case 8:
      {
        if (approximate_sync_) {
          ts_a_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], *filters_[4],
            *filters_[5], *filters_[6], *filters_[7]);
        } else {
          ts_e_->connectInput(
            *filters_[0], *filters_[1], *filters_[2], *filters_[3], *filters_[4],
            *filters_[5], *filters_[6], *filters_[7]);
        }
        break;
      }
    default:
      {
        NODELET_FATAL("Invalid 'input_topics' parameter given!");
        return;
      }
  }

  if (approximate_sync_) {
    ts_a_->registerCallback(
      boost::bind(
        &PointCloudConcatenateDataSynchronizer::input, this, _1, _2,
        _3, _4, _5, _6, _7, _8));
  } else {
    ts_e_->registerCallback(
      boost::bind(
        &PointCloudConcatenateDataSynchronizer::input, this, _1, _2,
        _3, _4, _5, _6, _7, _8));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PointCloudConcatenateDataSynchronizer::unsubscribe()
{
  for (int d = 0; d < filters_.size(); ++d) {
    filters_[d]->unsubscribe();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PointCloudConcatenateDataSynchronizer::combineClouds(
  const PointCloud2 & in1,
  const PointCloud2 & in2,
  PointCloud2 & out)
{
  // ROS_INFO ("Two pointclouds received: %zu and %zu.", in1.data.size (), in2.data.size ());
  PointCloud2::Ptr in1_t(new PointCloud2());
  PointCloud2::Ptr in2_t(new PointCloud2());

  // Transform the point clouds into the specified output frame
  if (output_frame_ != in1.header.frame_id) {
    pcl_ros::transformPointCloud(output_frame_, in1, *in1_t, tf_);
  } else {
    in1_t = boost::make_shared<PointCloud2>(in1);
  }

  if (output_frame_ != in2.header.frame_id) {
    pcl_ros::transformPointCloud(output_frame_, in2, *in2_t, tf_);
  } else {
    in2_t = boost::make_shared<PointCloud2>(in2);
  }

  // Concatenate the results
  pcl::concatenatePointCloud(*in1_t, *in2_t, out);
  // Copy header
  out.header.stamp = in1.header.stamp;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PointCloudConcatenateDataSynchronizer::input(
  const PointCloud2::ConstPtr & in1, const PointCloud2::ConstPtr & in2,
  const PointCloud2::ConstPtr & in3, const PointCloud2::ConstPtr & in4,
  const PointCloud2::ConstPtr & in5, const PointCloud2::ConstPtr & in6,
  const PointCloud2::ConstPtr & in7, const PointCloud2::ConstPtr & in8)
{
  PointCloud2::Ptr out1(new PointCloud2());
  PointCloud2::Ptr out2(new PointCloud2());
  pcl_ros::PointCloudConcatenateDataSynchronizer::combineClouds(*in1, *in2, *out1);
  if (in3 && in3->width * in3->height > 0) {
    pcl_ros::PointCloudConcatenateDataSynchronizer::combineClouds(*out1, *in3, *out2);
    out1 = out2;
    if (in4 && in4->width * in4->height > 0) {
      pcl_ros::PointCloudConcatenateDataSynchronizer::combineClouds(*out2, *in4, *out1);
      if (in5 && in5->width * in5->height > 0) {
        pcl_ros::PointCloudConcatenateDataSynchronizer::combineClouds(*out1, *in5, *out2);
        out1 = out2;
        if (in6 && in6->width * in6->height > 0) {
          pcl_ros::PointCloudConcatenateDataSynchronizer::combineClouds(*out2, *in6, *out1);
          if (in7 && in7->width * in7->height > 0) {
            pcl_ros::PointCloudConcatenateDataSynchronizer::combineClouds(*out1, *in7, *out2);
            out1 = out2;
          }
        }
      }
    }
  }
  pub_output_.publish(boost::make_shared<PointCloud2>(*out1));
}

typedef pcl_ros::PointCloudConcatenateDataSynchronizer PointCloudConcatenateDataSynchronizer;
PLUGINLIB_EXPORT_CLASS(PointCloudConcatenateDataSynchronizer, nodelet::Nodelet);
