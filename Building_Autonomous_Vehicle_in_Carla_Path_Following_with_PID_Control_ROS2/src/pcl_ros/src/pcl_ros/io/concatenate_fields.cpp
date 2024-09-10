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
 * $Id: concatenate_fields.cpp 35052 2011-01-03 21:04:57Z rusu $
 *
 */

/** \author Radu Bogdan Rusu */

#include <pluginlib/class_list_macros.h>
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include "pcl_ros/io/concatenate_fields.hpp"


///////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PointCloudConcatenateFieldsSynchronizer::onInit()
{
  nodelet_topic_tools::NodeletLazy::onInit();

  // ---[ Mandatory parameters
  if (!pnh_->getParam("input_messages", input_messages_)) {
    NODELET_ERROR("[onInit] Need a 'input_messages' parameter to be set before continuing!");
    return;
  }
  if (input_messages_ < 2) {
    NODELET_ERROR("[onInit] Invalid 'input_messages' parameter given!");
    return;
  }
  // ---[ Optional parameters
  pnh_->getParam("max_queue_size", maximum_queue_size_);
  pnh_->getParam("maximum_seconds", maximum_seconds_);
  pub_output_ = advertise<sensor_msgs::PointCloud2>(*pnh_, "output", maximum_queue_size_);

  onInitPostProcess();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PointCloudConcatenateFieldsSynchronizer::subscribe()
{
  sub_input_ = pnh_->subscribe(
    "input", maximum_queue_size_,
    &PointCloudConcatenateFieldsSynchronizer::input_callback, this);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PointCloudConcatenateFieldsSynchronizer::unsubscribe()
{
  sub_input_.shutdown();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::PointCloudConcatenateFieldsSynchronizer::input_callback(const PointCloudConstPtr & cloud)
{
  NODELET_DEBUG(
    "[input_callback] PointCloud with %d data points (%s), stamp %f, and frame %s on "
    "topic %s received.",
    cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(),
    cloud->header.stamp.toSec(), cloud->header.frame_id.c_str(),
    pnh_->resolveName("input").c_str());

  // Erase old data in the queue
  if (maximum_seconds_ > 0 && queue_.size() > 0) {
    while (fabs( ( (*queue_.begin()).first - cloud->header.stamp).toSec() ) > maximum_seconds_ &&
      queue_.size() > 0)
    {
      NODELET_WARN(
        "[input_callback] Maximum seconds limit (%f) reached. Difference is %f, erasing message "
        "in queue with stamp %f.", maximum_seconds_,
        (*queue_.begin()).first.toSec(),
        fabs( ( (*queue_.begin()).first - cloud->header.stamp).toSec() ));
      queue_.erase(queue_.begin());
    }
  }

  // Push back new data
  queue_[cloud->header.stamp].push_back(cloud);
  if (static_cast<int>(queue_[cloud->header.stamp].size()) >= input_messages_) {
    // Concatenate together and publish
    std::vector<PointCloudConstPtr> & clouds = queue_[cloud->header.stamp];
    PointCloud cloud_out = *clouds[0];

    // Resize the output dataset
    int data_size = cloud_out.data.size();
    int nr_fields = cloud_out.fields.size();
    int nr_points = cloud_out.width * cloud_out.height;
    for (size_t i = 1; i < clouds.size(); ++i) {
      assert(
        clouds[i]->data.size() / (clouds[i]->width * clouds[i]->height) == clouds[i]->point_step);

      if (clouds[i]->width != cloud_out.width || clouds[i]->height != cloud_out.height) {
        NODELET_ERROR(
          "[input_callback] Width/height of pointcloud %zu (%dx%d) differs "
          "from the others (%dx%d)!",
          i, clouds[i]->width, clouds[i]->height, cloud_out.width, cloud_out.height);
        break;
      }
      // Point step must increase with the length of each new field
      cloud_out.point_step += clouds[i]->point_step;
      // Resize data to hold all clouds
      data_size += clouds[i]->data.size();

      // Concatenate fields
      cloud_out.fields.resize(nr_fields + clouds[i]->fields.size());
      int delta_offset = cloud_out.fields[nr_fields - 1].offset + pcl::getFieldSize(
        cloud_out.fields[nr_fields - 1].datatype);
      for (size_t d = 0; d < clouds[i]->fields.size(); ++d) {
        cloud_out.fields[nr_fields + d] = clouds[i]->fields[d];
        cloud_out.fields[nr_fields + d].offset += delta_offset;
      }
      nr_fields += clouds[i]->fields.size();
    }
    // Recalculate row_step
    cloud_out.row_step = cloud_out.point_step * cloud_out.width;
    cloud_out.data.resize(data_size);

    // Iterate over each point and perform the appropriate memcpys
    int point_offset = 0;
    for (int cp = 0; cp < nr_points; ++cp) {
      for (size_t i = 0; i < clouds.size(); ++i) {
        // Copy each individual point
        memcpy(
          &cloud_out.data[point_offset], &clouds[i]->data[cp * clouds[i]->point_step],
          clouds[i]->point_step);
        point_offset += clouds[i]->point_step;
      }
    }
    pub_output_.publish(boost::make_shared<const PointCloud>(cloud_out));
    queue_.erase(cloud->header.stamp);
  }

  // Clean the queue to avoid overflowing
  if (maximum_queue_size_ > 0) {
    while (static_cast<int>(queue_.size()) > maximum_queue_size_) {
      queue_.erase(queue_.begin());
    }
  }
}

typedef pcl_ros::PointCloudConcatenateFieldsSynchronizer PointCloudConcatenateFieldsSynchronizer;
PLUGINLIB_EXPORT_CLASS(PointCloudConcatenateFieldsSynchronizer, nodelet::Nodelet);
