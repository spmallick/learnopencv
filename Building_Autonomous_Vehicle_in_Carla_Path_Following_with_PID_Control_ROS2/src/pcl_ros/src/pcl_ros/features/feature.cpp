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
 *  COPYRIGHT OWNERff OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: feature.cpp 35422 2011-01-24 20:04:44Z rusu $
 *
 */

// #include <pluginlib/class_list_macros.h>
// Include the implementations here instead of compiling them separately to speed up compile time
// #include "normal_3d.cpp"
// #include "boundary.cpp"
// #include "fpfh.cpp"
// #include "fpfh_omp.cpp"
// #include "moment_invariants.cpp"
// #include "normal_3d_omp.cpp"
// #include "normal_3d_tbb.cpp"
// #include "pfh.cpp"
// #include "principal_curvatures.cpp"
// #include "vfh.cpp"
#include <pcl/common/io.h>
#include <message_filters/null_types.h>
#include <vector>
#include "pcl_ros/features/feature.hpp"

////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::Feature::onInit()
{
  // Call the super onInit ()
  PCLNodelet::onInit();

  // Call the child init
  childInit(*pnh_);

  // Allow each individual class that inherits from us to declare their own Publisher
  // This is useful for Publisher<pcl::PointCloud<T> >, as NormalEstimation can publish
  // PointCloud<Normal>, etc
  // pub_output_ = pnh_->template advertise<PointCloud2> ("output", max_queue_size_);

  // ---[ Mandatory parameters
  if (!pnh_->getParam("k_search", k_) && !pnh_->getParam("radius_search", search_radius_)) {
    NODELET_ERROR(
      "[%s::onInit] Neither 'k_search' nor 'radius_search' set! Need to set at least one of these "
      "parameters before continuing.",
      getName().c_str());
    return;
  }
  if (!pnh_->getParam("spatial_locator", spatial_locator_type_)) {
    NODELET_ERROR(
      "[%s::onInit] Need a 'spatial_locator' parameter to be set before continuing!",
      getName().c_str());
    return;
  }

  // ---[ Optional parameters
  pnh_->getParam("use_surface", use_surface_);

  // Enable the dynamic reconfigure service
  srv_ = boost::make_shared<dynamic_reconfigure::Server<FeatureConfig>>(*pnh_);
  dynamic_reconfigure::Server<FeatureConfig>::CallbackType f = boost::bind(
    &Feature::config_callback, this, _1, _2);
  srv_->setCallback(f);

  NODELET_DEBUG(
    "[%s::onInit] Nodelet successfully created with the following parameters:\n"
    " - use_surface    : %s\n"
    " - k_search       : %d\n"
    " - radius_search  : %f\n"
    " - spatial_locator: %d",
    getName().c_str(),
    (use_surface_) ? "true" : "false", k_, search_radius_, spatial_locator_type_);

  onInitPostProcess();
}

////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::Feature::subscribe()
{
  // If we're supposed to look for PointIndices (indices) or PointCloud (surface) messages
  if (use_indices_ || use_surface_) {
    // Create the objects here
    if (approximate_sync_) {
      sync_input_surface_indices_a_ =
        boost::make_shared<message_filters::Synchronizer<
            sync_policies::ApproximateTime<
              PointCloudIn, PointCloudIn, PointIndices>>>(max_queue_size_);
    } else {
      sync_input_surface_indices_e_ =
        boost::make_shared<message_filters::Synchronizer<
            sync_policies::ExactTime<PointCloudIn, PointCloudIn, PointIndices>>>(max_queue_size_);
    }

    // Subscribe to the input using a filter
    sub_input_filter_.subscribe(*pnh_, "input", max_queue_size_);
    if (use_indices_) {
      // If indices are enabled, subscribe to the indices
      sub_indices_filter_.subscribe(*pnh_, "indices", max_queue_size_);
      if (use_surface_) {   // Use both indices and surface
        // If surface is enabled, subscribe to the surface,
        // connect the input-indices-surface trio and register
        sub_surface_filter_.subscribe(*pnh_, "surface", max_queue_size_);
        if (approximate_sync_) {
          sync_input_surface_indices_a_->connectInput(
            sub_input_filter_, sub_surface_filter_,
            sub_indices_filter_);
        } else {
          sync_input_surface_indices_e_->connectInput(
            sub_input_filter_, sub_surface_filter_,
            sub_indices_filter_);
        }
      } else {              // Use only indices
        sub_input_filter_.registerCallback(bind(&Feature::input_callback, this, _1));
        // surface not enabled, connect the input-indices duo and register
        if (approximate_sync_) {
          sync_input_surface_indices_a_->connectInput(
            sub_input_filter_, nf_pc_,
            sub_indices_filter_);
        } else {
          sync_input_surface_indices_e_->connectInput(
            sub_input_filter_, nf_pc_,
            sub_indices_filter_);
        }
      }
    } else {                // Use only surface
      sub_input_filter_.registerCallback(bind(&Feature::input_callback, this, _1));
      // indices not enabled, connect the input-surface duo and register
      sub_surface_filter_.subscribe(*pnh_, "surface", max_queue_size_);
      if (approximate_sync_) {
        sync_input_surface_indices_a_->connectInput(sub_input_filter_, sub_surface_filter_, nf_pi_);
      } else {
        sync_input_surface_indices_e_->connectInput(sub_input_filter_, sub_surface_filter_, nf_pi_);
      }
    }
    // Register callbacks
    if (approximate_sync_) {
      sync_input_surface_indices_a_->registerCallback(
        bind(
          &Feature::input_surface_indices_callback,
          this, _1, _2, _3));
    } else {
      sync_input_surface_indices_e_->registerCallback(
        bind(
          &Feature::input_surface_indices_callback,
          this, _1, _2, _3));
    }
  } else {
    // Subscribe in an old fashion to input only (no filters)
    sub_input_ =
      pnh_->subscribe<PointCloudIn>(
      "input", max_queue_size_,
      bind(
        &Feature::input_surface_indices_callback, this, _1, PointCloudInConstPtr(),
        PointIndicesConstPtr()));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::Feature::unsubscribe()
{
  if (use_indices_ || use_surface_) {
    sub_input_filter_.unsubscribe();
    if (use_indices_) {
      sub_indices_filter_.unsubscribe();
      if (use_surface_) {
        sub_surface_filter_.unsubscribe();
      }
    } else {
      sub_surface_filter_.unsubscribe();
    }
  } else {
    sub_input_.shutdown();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::Feature::config_callback(FeatureConfig & config, uint32_t level)
{
  if (k_ != config.k_search) {
    k_ = config.k_search;
    NODELET_DEBUG(
      "[config_callback] Setting the number of K nearest neighbors to use for each point: %d.", k_);
  }
  if (search_radius_ != config.radius_search) {
    search_radius_ = config.radius_search;
    NODELET_DEBUG(
      "[config_callback] Setting the nearest neighbors search radius for each point: %f.",
      search_radius_);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::Feature::input_surface_indices_callback(
  const PointCloudInConstPtr & cloud,
  const PointCloudInConstPtr & cloud_surface, const PointIndicesConstPtr & indices)
{
  // No subscribers, no work
  if (pub_output_.getNumSubscribers() <= 0) {
    return;
  }

  // If cloud is given, check if it's valid
  if (!isValid(cloud)) {
    NODELET_ERROR("[%s::input_surface_indices_callback] Invalid input!", getName().c_str());
    emptyPublish(cloud);
    return;
  }

  // If surface is given, check if it's valid
  if (cloud_surface && !isValid(cloud_surface, "surface")) {
    NODELET_ERROR("[%s::input_surface_indices_callback] Invalid input surface!", getName().c_str());
    emptyPublish(cloud);
    return;
  }

  // If indices are given, check if they are valid
  if (indices && !isValid(indices)) {
    NODELET_ERROR("[%s::input_surface_indices_callback] Invalid input indices!", getName().c_str());
    emptyPublish(cloud);
    return;
  }

  /// DEBUG
  if (cloud_surface) {
    if (indices) {
      NODELET_DEBUG(
        "[input_surface_indices_callback]\n"
        "                                         - PointCloud with %d data points (%s), stamp %f, "
        "and frame %s on topic %s received.\n"
        "                                         - PointCloud with %d data points (%s), stamp %f, "
        "and frame %s on topic %s received.\n"
        "                                         - PointIndices with %zu values, stamp %f, and "
        "frame %s on topic %s received.",
        cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
          cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
          "input").c_str(),
        cloud_surface->width * cloud_surface->height, pcl::getFieldsList(
          *cloud_surface).c_str(), fromPCL(
          cloud_surface->header).stamp.toSec(),
        cloud_surface->header.frame_id.c_str(), pnh_->resolveName("surface").c_str(),
        indices->indices.size(), indices->header.stamp.toSec(),
        indices->header.frame_id.c_str(), pnh_->resolveName("indices").c_str());
    } else {
      NODELET_DEBUG(
        "[input_surface_indices_callback]\n"
        "                                 - PointCloud with %d data points (%s), stamp %f, and "
        "frame %s on topic %s received.\n"
        "                                 - PointCloud with %d data points (%s), stamp %f, and "
        "frame %s on topic %s received.",
        cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
          cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
          "input").c_str(),
        cloud_surface->width * cloud_surface->height, pcl::getFieldsList(
          *cloud_surface).c_str(), fromPCL(
          cloud_surface->header).stamp.toSec(),
        cloud_surface->header.frame_id.c_str(), pnh_->resolveName("surface").c_str());
    }
  } else if (indices) {
    NODELET_DEBUG(
      "[input_surface_indices_callback]\n"
      "                                 - PointCloud with %d data points (%s), stamp %f, and frame "
      "%s on topic %s received.\n"
      "                                 - PointIndices with %zu values, stamp %f, and frame %s on "
      "topic %s received.",
      cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
        "input").c_str(),
      indices->indices.size(), indices->header.stamp.toSec(),
      indices->header.frame_id.c_str(), pnh_->resolveName("indices").c_str());
  } else {
    NODELET_DEBUG(
      "[input_surface_indices_callback] PointCloud with %d data points, stamp %f, and frame %s on "
      "topic %s received.", cloud->width * cloud->height, fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
        "input").c_str());
  }
  ///


  if (static_cast<int>(cloud->width * cloud->height) < k_) {
    NODELET_ERROR(
      "[input_surface_indices_callback] Requested number of k-nearest neighbors (%d) is larger "
      "than the PointCloud size (%d)!", k_,
      (int)(cloud->width * cloud->height));
    emptyPublish(cloud);
    return;
  }

  // If indices given...
  IndicesPtr vindices;
  if (indices && !indices->header.frame_id.empty()) {
    vindices.reset(new std::vector<int>(indices->indices));
  }

  computePublish(cloud, cloud_surface, vindices);
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::FeatureFromNormals::onInit()
{
  // Call the super onInit ()
  PCLNodelet::onInit();

  // Call the child init
  childInit(*pnh_);

  // Allow each individual class that inherits from us to declare their own Publisher
  // This is useful for Publisher<pcl::PointCloud<T> >, as NormalEstimation can publish
  // PointCloud<Normal>, etc
  // pub_output_ = pnh_->template advertise<PointCloud2> ("output", max_queue_size_);

  // ---[ Mandatory parameters
  if (!pnh_->getParam("k_search", k_) && !pnh_->getParam("radius_search", search_radius_)) {
    NODELET_ERROR(
      "[%s::onInit] Neither 'k_search' nor 'radius_search' set! Need to set at least one of these "
      "parameters before continuing.",
      getName().c_str());
    return;
  }
  if (!pnh_->getParam("spatial_locator", spatial_locator_type_)) {
    NODELET_ERROR(
      "[%s::onInit] Need a 'spatial_locator' parameter to be set before continuing!",
      getName().c_str());
    return;
  }
  // ---[ Optional parameters
  pnh_->getParam("use_surface", use_surface_);

  // Enable the dynamic reconfigure service
  srv_ = boost::make_shared<dynamic_reconfigure::Server<FeatureConfig>>(*pnh_);
  dynamic_reconfigure::Server<FeatureConfig>::CallbackType f = boost::bind(
    &FeatureFromNormals::config_callback, this, _1, _2);
  srv_->setCallback(f);

  NODELET_DEBUG(
    "[%s::onInit] Nodelet successfully created with the following parameters:\n"
    " - use_surface    : %s\n"
    " - k_search       : %d\n"
    " - radius_search  : %f\n"
    " - spatial_locator: %d",
    getName().c_str(),
    (use_surface_) ? "true" : "false", k_, search_radius_, spatial_locator_type_);

  onInitPostProcess();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::FeatureFromNormals::subscribe()
{
  sub_input_filter_.subscribe(*pnh_, "input", max_queue_size_);
  sub_normals_filter_.subscribe(*pnh_, "normals", max_queue_size_);

  // Create the objects here
  if (approximate_sync_) {
    sync_input_normals_surface_indices_a_ =
      boost::make_shared<message_filters::Synchronizer<
          sync_policies::ApproximateTime<PointCloudIn, PointCloudN,
          PointCloudIn, PointIndices>>>(max_queue_size_);
  } else {
    sync_input_normals_surface_indices_e_ =
      boost::make_shared<message_filters::Synchronizer<sync_policies::ExactTime<PointCloudIn,
        PointCloudN, PointCloudIn, PointIndices>>>(max_queue_size_);
  }

  // If we're supposed to look for PointIndices (indices) or PointCloud (surface) messages
  if (use_indices_ || use_surface_) {
    if (use_indices_) {
      // If indices are enabled, subscribe to the indices
      sub_indices_filter_.subscribe(*pnh_, "indices", max_queue_size_);
      if (use_surface_) {   // Use both indices and surface
        // If surface is enabled, subscribe to the surface, connect the input-indices-surface trio
        // and register
        sub_surface_filter_.subscribe(*pnh_, "surface", max_queue_size_);
        if (approximate_sync_) {
          sync_input_normals_surface_indices_a_->connectInput(
            sub_input_filter_,
            sub_normals_filter_,
            sub_surface_filter_,
            sub_indices_filter_);
        } else {
          sync_input_normals_surface_indices_e_->connectInput(
            sub_input_filter_,
            sub_normals_filter_,
            sub_surface_filter_,
            sub_indices_filter_);
        }
      } else {              // Use only indices
        sub_input_filter_.registerCallback(bind(&FeatureFromNormals::input_callback, this, _1));
        if (approximate_sync_) {
          // surface not enabled, connect the input-indices duo and register
          sync_input_normals_surface_indices_a_->connectInput(
            sub_input_filter_,
            sub_normals_filter_, nf_pc_,
            sub_indices_filter_);
        } else {
          // surface not enabled, connect the input-indices duo and register
          sync_input_normals_surface_indices_e_->connectInput(
            sub_input_filter_,
            sub_normals_filter_, nf_pc_,
            sub_indices_filter_);
        }
      }
    } else {                // Use only surface
      // indices not enabled, connect the input-surface duo and register
      sub_surface_filter_.subscribe(*pnh_, "surface", max_queue_size_);

      sub_input_filter_.registerCallback(bind(&FeatureFromNormals::input_callback, this, _1));
      if (approximate_sync_) {
        sync_input_normals_surface_indices_a_->connectInput(
          sub_input_filter_, sub_normals_filter_,
          sub_surface_filter_, nf_pi_);
      } else {
        sync_input_normals_surface_indices_e_->connectInput(
          sub_input_filter_, sub_normals_filter_,
          sub_surface_filter_, nf_pi_);
      }
    }
  } else {
    sub_input_filter_.registerCallback(bind(&FeatureFromNormals::input_callback, this, _1));

    if (approximate_sync_) {
      sync_input_normals_surface_indices_a_->connectInput(
        sub_input_filter_, sub_normals_filter_,
        nf_pc_, nf_pi_);
    } else {
      sync_input_normals_surface_indices_e_->connectInput(
        sub_input_filter_, sub_normals_filter_,
        nf_pc_, nf_pi_);
    }
  }

  // Register callbacks
  if (approximate_sync_) {
    sync_input_normals_surface_indices_a_->registerCallback(
      bind(
        &FeatureFromNormals::
        input_normals_surface_indices_callback, this, _1, _2, _3, _4));
  } else {
    sync_input_normals_surface_indices_e_->registerCallback(
      bind(
        &FeatureFromNormals::
        input_normals_surface_indices_callback, this, _1, _2, _3, _4));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::FeatureFromNormals::unsubscribe()
{
  sub_input_filter_.unsubscribe();
  sub_normals_filter_.unsubscribe();
  if (use_indices_ || use_surface_) {
    if (use_indices_) {
      sub_indices_filter_.unsubscribe();
      if (use_surface_) {
        sub_surface_filter_.unsubscribe();
      }
    } else {
      sub_surface_filter_.unsubscribe();
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void
pcl_ros::FeatureFromNormals::input_normals_surface_indices_callback(
  const PointCloudInConstPtr & cloud, const PointCloudNConstPtr & cloud_normals,
  const PointCloudInConstPtr & cloud_surface, const PointIndicesConstPtr & indices)
{
  // No subscribers, no work
  if (pub_output_.getNumSubscribers() <= 0) {
    return;
  }

  // If cloud+normals is given, check if it's valid
  if (!isValid(cloud)) {  // || !isValid (cloud_normals, "normals"))
    NODELET_ERROR("[%s::input_normals_surface_indices_callback] Invalid input!", getName().c_str());
    emptyPublish(cloud);
    return;
  }

  // If surface is given, check if it's valid
  if (cloud_surface && !isValid(cloud_surface, "surface")) {
    NODELET_ERROR(
      "[%s::input_normals_surface_indices_callback] Invalid input surface!",
      getName().c_str());
    emptyPublish(cloud);
    return;
  }

  // If indices are given, check if they are valid
  if (indices && !isValid(indices)) {
    NODELET_ERROR(
      "[%s::input_normals_surface_indices_callback] Invalid input indices!",
      getName().c_str());
    emptyPublish(cloud);
    return;
  }

  /// DEBUG
  if (cloud_surface) {
    if (indices) {
      NODELET_DEBUG(
        "[%s::input_normals_surface_indices_callback]\n"
        "                                                 - PointCloud with %d data points (%s), "
        "stamp %f, and frame %s on topic %s received.\n"
        "                                                 - PointCloud with %d data points (%s), "
        "stamp %f, and frame %s on topic %s received.\n"
        "                                                 - PointCloud with %d data points (%s), "
        "stamp %f, and frame %s on topic %s received.\n"
        "                                                 - PointIndices with %zu values, "
        "stamp %f, and frame %s on topic %s received.",
        getName().c_str(),
        cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
          cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
          "input").c_str(),
        cloud_surface->width * cloud_surface->height, pcl::getFieldsList(
          *cloud_surface).c_str(), fromPCL(
          cloud_surface->header).stamp.toSec(),
        cloud_surface->header.frame_id.c_str(), pnh_->resolveName("surface").c_str(),
        cloud_normals->width * cloud_normals->height, pcl::getFieldsList(
          *cloud_normals).c_str(), fromPCL(
          cloud_normals->header).stamp.toSec(),
        cloud_normals->header.frame_id.c_str(), pnh_->resolveName("normals").c_str(),
        indices->indices.size(), indices->header.stamp.toSec(),
        indices->header.frame_id.c_str(), pnh_->resolveName("indices").c_str());
    } else {
      NODELET_DEBUG(
        "[%s::input_normals_surface_indices_callback]\n"
        "                                         - PointCloud with %d data points (%s), "
        "stamp %f, and frame %s on topic %s received.\n"
        "                                         - PointCloud with %d data points (%s), "
        "stamp %f, and frame %s on topic %s received.\n"
        "                                         - PointCloud with %d data points (%s), "
        "stamp %f, and frame %s on topic %s received.",
        getName().c_str(),
        cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
          cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
          "input").c_str(),
        cloud_surface->width * cloud_surface->height, pcl::getFieldsList(
          *cloud_surface).c_str(), fromPCL(
          cloud_surface->header).stamp.toSec(),
        cloud_surface->header.frame_id.c_str(), pnh_->resolveName("surface").c_str(),
        cloud_normals->width * cloud_normals->height, pcl::getFieldsList(
          *cloud_normals).c_str(), fromPCL(
          cloud_normals->header).stamp.toSec(),
        cloud_normals->header.frame_id.c_str(), pnh_->resolveName("normals").c_str());
    }
  } else if (indices) {
    NODELET_DEBUG(
      "[%s::input_normals_surface_indices_callback]\n"
      "                                         - PointCloud with %d data points (%s), "
      "stamp %f, and frame %s on topic %s received.\n"
      "                                         - PointCloud with %d data points (%s), "
      "stamp %f, and frame %s on topic %s received.\n"
      "                                         - PointIndices with %zu values, "
      "stamp %f, and frame %s on topic %s received.",
      getName().c_str(),
      cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
        "input").c_str(),
      cloud_normals->width * cloud_normals->height, pcl::getFieldsList(
        *cloud_normals).c_str(), fromPCL(
        cloud_normals->header).stamp.toSec(),
      cloud_normals->header.frame_id.c_str(), pnh_->resolveName("normals").c_str(),
      indices->indices.size(), indices->header.stamp.toSec(),
      indices->header.frame_id.c_str(), pnh_->resolveName("indices").c_str());
  } else {
    NODELET_DEBUG(
      "[%s::input_normals_surface_indices_callback]\n"
      "                                 - PointCloud with %d data points (%s), stamp %f, and "
      "frame %s on topic %s received.\n"
      "                                 - PointCloud with %d data points (%s), stamp %f, and "
      "frame %s on topic %s received.",
      getName().c_str(),
      cloud->width * cloud->height, pcl::getFieldsList(*cloud).c_str(), fromPCL(
        cloud->header).stamp.toSec(), cloud->header.frame_id.c_str(), pnh_->resolveName(
        "input").c_str(),
      cloud_normals->width * cloud_normals->height, pcl::getFieldsList(
        *cloud_normals).c_str(), fromPCL(
        cloud_normals->header).stamp.toSec(),
      cloud_normals->header.frame_id.c_str(), pnh_->resolveName("normals").c_str());
  }
  ///

  if (static_cast<int>(cloud->width * cloud->height) < k_) {
    NODELET_ERROR(
      "[%s::input_normals_surface_indices_callback] Requested number of k-nearest neighbors (%d) "
      "is larger than the PointCloud size (%d)!",
      getName().c_str(), k_, (int)(cloud->width * cloud->height));
    emptyPublish(cloud);
    return;
  }

  // If indices given...
  IndicesPtr vindices;
  if (indices && !indices->header.frame_id.empty()) {
    vindices.reset(new std::vector<int>(indices->indices));
  }

  computePublish(cloud, cloud_normals, cloud_surface, vindices);
}
