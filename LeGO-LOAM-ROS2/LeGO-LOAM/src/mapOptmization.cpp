// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar
//   Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems
//      (IROS). October 2018.

#include "mapOptimization.h"
#include <future>

using namespace gtsam;

const std::string PARAM_ENABLE_LOOP = "mapping.enable_loop_closure";
const std::string PARAM_SEARCH_RADIUS = "mapping.surrounding_keyframe_search_radius";
const std::string PARAM_SEARCH_NUM = "mapping.surrounding_keyframe_search_num";
const std::string PARAM_HISTORY_SEARCH_RADIUS = "mapping.history_keyframe_search_radius";
const std::string PARAM_HISTORY_SEARCH_NUM = "mapping.history_keyframe_search_num";
const std::string PARAM_HISTORY_SCORE = "mapping.history_keyframe_fitness_score";
const std::string PARAM_GLOBAL_SEARCH_RADIUS = "mapping.global_map_visualization_search_radius";

MapOptimization::MapOptimization(const std::string &name, Channel<AssociationOut> &input_channel)
    : Node(name), _input_channel(input_channel), _publish_global_signal(false), _loop_closure_signal(false) {
  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  isam = new ISAM2(parameters);

  pubKeyPoses = this->create_publisher<sensor_msgs::msg::PointCloud2>("/key_pose_origin", 2);
  pubLaserCloudSurround = this->create_publisher<sensor_msgs::msg::PointCloud2>("/laser_cloud_surround", 2);
  pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry>("/aft_mapped_to_init", 5);
  pubHistoryKeyFrames = this->create_publisher<sensor_msgs::msg::PointCloud2>("/history_cloud", 2);
  pubIcpKeyFrames = this->create_publisher<sensor_msgs::msg::PointCloud2>("/corrected_cloud", 2);
  pubRecentKeyFrames = this->create_publisher<sensor_msgs::msg::PointCloud2>("/recent_cloud", 2);

  tfBroadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
  downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
  downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

  // for histor key frames of loop closure
  downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4);
  // for surrounding key poses of scan-to-map optimization
  downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0);

  // for global map visualization
  downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0);
  // for global map visualization
  downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4);

  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";

  aftMappedTrans.header.frame_id = "camera_init";
  aftMappedTrans.child_frame_id = "aft_mapped";

  // Declare parameters
  this->declare_parameter(PARAM_ENABLE_LOOP,_loop_closure_enabled);
  this->declare_parameter(PARAM_SEARCH_RADIUS,_surrounding_keyframe_search_radius);
  this->declare_parameter(PARAM_SEARCH_NUM,_surrounding_keyframe_search_num);
  this->declare_parameter(PARAM_HISTORY_SEARCH_RADIUS,_history_keyframe_search_radius);
  this->declare_parameter(PARAM_HISTORY_SEARCH_NUM,_history_keyframe_search_num);
  this->declare_parameter(PARAM_HISTORY_SCORE,_history_keyframe_fitness_score);
  this->declare_parameter(PARAM_GLOBAL_SEARCH_RADIUS,_global_map_visualization_search_radius);

  // Read parameters
  if (!this->get_parameter(PARAM_ENABLE_LOOP, _loop_closure_enabled)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_ENABLE_LOOP.c_str());
  }
  if (!this->get_parameter(PARAM_SEARCH_RADIUS, _surrounding_keyframe_search_radius)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_SEARCH_RADIUS.c_str());
  }
  if (!this->get_parameter(PARAM_SEARCH_NUM, _surrounding_keyframe_search_num)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_SEARCH_NUM.c_str());
  }
  if (!this->get_parameter(PARAM_HISTORY_SEARCH_RADIUS, _history_keyframe_search_radius)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_HISTORY_SEARCH_RADIUS.c_str());
  }
  if (!this->get_parameter(PARAM_HISTORY_SEARCH_NUM, _history_keyframe_search_num)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_HISTORY_SEARCH_NUM.c_str());
  }
  if (!this->get_parameter(PARAM_HISTORY_SCORE, _history_keyframe_fitness_score)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_HISTORY_SCORE.c_str());
  }
  if (!this->get_parameter(PARAM_GLOBAL_SEARCH_RADIUS, _global_map_visualization_search_radius)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_GLOBAL_SEARCH_RADIUS.c_str());
  }

  allocateMemory();

  _publish_global_thread = std::thread(&MapOptimization::publishGlobalMapThread, this);
  _loop_closure_thread = std::thread(&MapOptimization::loopClosureThread, this);
  _run_thread = std::thread(&MapOptimization::run, this);

}

MapOptimization::~MapOptimization()
{
  _input_channel.send({});
  _run_thread.join();

  _publish_global_signal.send(false);
  _publish_global_thread.join();

  _loop_closure_signal.send(false);
  _loop_closure_thread.join();
}

void MapOptimization::allocateMemory() {
  cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
  cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

  surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
  surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());

  laserCloudCornerLast.reset(
      new pcl::PointCloud<PointType>());  // corner feature set from
                                          // odoOptimization
  laserCloudSurfLast.reset(
      new pcl::PointCloud<PointType>());  // surf feature set from
                                          // odoOptimization
  laserCloudCornerLastDS.reset(
      new pcl::PointCloud<PointType>());  // downsampled corner featuer set
                                          // from odoOptimization
  laserCloudSurfLastDS.reset(
      new pcl::PointCloud<PointType>());  // downsampled surf featuer set from
                                          // odoOptimization
  laserCloudOutlierLast.reset(
      new pcl::PointCloud<PointType>());  // corner feature set from
                                          // odoOptimization
  laserCloudOutlierLastDS.reset(
      new pcl::PointCloud<PointType>());  // downsampled corner feature set
                                          // from odoOptimization
  laserCloudSurfTotalLast.reset(
      new pcl::PointCloud<PointType>());  // surf feature set from
                                          // odoOptimization
  laserCloudSurfTotalLastDS.reset(
      new pcl::PointCloud<PointType>());  // downsampled surf featuer set from
                                          // odoOptimization

  laserCloudOri.reset(new pcl::PointCloud<PointType>());
  coeffSel.reset(new pcl::PointCloud<PointType>());

  laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
  laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
  laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

  nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
  nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
  nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
  nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

  latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
  latestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
  latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

  globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
  globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
  globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
  globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

  timeLaserOdometry = this->now();

  for (int i = 0; i < 6; ++i) {
    transformLast[i] = 0;
    transformSum[i] = 0;
    transformIncre[i] = 0;
    transformTobeMapped[i] = 0;
    transformBefMapped[i] = 0;
    transformAftMapped[i] = 0;
  }


  matA0.setZero();
  matB0.fill(-1);
  matX0.setZero();

  matA1.setZero();
  matD1.setZero();
  matV1.setZero();

  isDegenerate = false;
  matP.setZero();

  laserCloudCornerFromMapDSNum = 0;
  laserCloudSurfFromMapDSNum = 0;
  laserCloudCornerLastDSNum = 0;
  laserCloudSurfLastDSNum = 0;
  laserCloudOutlierLastDSNum = 0;
  laserCloudSurfTotalLastDSNum = 0;

  potentialLoopFlag = false;
  aLoopIsClosed = false;

  latestFrameID = 0;
}


void MapOptimization::publishGlobalMapThread()
{
  while(rclcpp::ok())
  {
    bool ready;
    _publish_global_signal.receive(ready);
    if(ready){
      publishGlobalMap();
    }
  }
}

void MapOptimization::loopClosureThread()
{
  while(rclcpp::ok())
  {
    bool ready;
    _loop_closure_signal.receive(ready);
    if(ready && _loop_closure_enabled){
      performLoopClosure();
    }
  }
}

void MapOptimization::transformAssociateToMap() {
  float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) -
             sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
  float y1 = transformBefMapped[4] - transformSum[4];
  float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) +
             cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

  float x2 = x1;
  float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
  float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

  transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
  transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
  transformIncre[5] = z2;

  float sbcx = sin(transformSum[0]);
  float cbcx = cos(transformSum[0]);
  float sbcy = sin(transformSum[1]);
  float cbcy = cos(transformSum[1]);
  float sbcz = sin(transformSum[2]);
  float cbcz = cos(transformSum[2]);

  float sblx = sin(transformBefMapped[0]);
  float cblx = cos(transformBefMapped[0]);
  float sbly = sin(transformBefMapped[1]);
  float cbly = cos(transformBefMapped[1]);
  float sblz = sin(transformBefMapped[2]);
  float cblz = cos(transformBefMapped[2]);

  float salx = sin(transformAftMapped[0]);
  float calx = cos(transformAftMapped[0]);
  float saly = sin(transformAftMapped[1]);
  float caly = cos(transformAftMapped[1]);
  float salz = sin(transformAftMapped[2]);
  float calz = cos(transformAftMapped[2]);

  float srx = -sbcx * (salx * sblx + calx * cblx * salz * sblz +
                       calx * calz * cblx * cblz) -
              cbcx * sbcy *
                  (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
                   calx * salz * (cbly * cblz + sblx * sbly * sblz) +
                   cblx * salx * sbly) -
              cbcx * cbcy *
                  (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
                   calx * calz * (sbly * sblz + cbly * cblz * sblx) +
                   cblx * cbly * salx);
  transformTobeMapped[0] = -asin(srx);

  float srycrx = sbcx * (cblx * cblz * (caly * salz - calz * salx * saly) -
                         cblx * sblz * (caly * calz + salx * saly * salz) +
                         calx * saly * sblx) -
                 cbcx * cbcy *
                     ((caly * calz + salx * saly * salz) *
                          (cblz * sbly - cbly * sblx * sblz) +
                      (caly * salz - calz * salx * saly) *
                          (sbly * sblz + cbly * cblz * sblx) -
                      calx * cblx * cbly * saly) +
                 cbcx * sbcy *
                     ((caly * calz + salx * saly * salz) *
                          (cbly * cblz + sblx * sbly * sblz) +
                      (caly * salz - calz * salx * saly) *
                          (cbly * sblz - cblz * sblx * sbly) +
                      calx * cblx * saly * sbly);
  float crycrx = sbcx * (cblx * sblz * (calz * saly - caly * salx * salz) -
                         cblx * cblz * (saly * salz + caly * calz * salx) +
                         calx * caly * sblx) +
                 cbcx * cbcy *
                     ((saly * salz + caly * calz * salx) *
                          (sbly * sblz + cbly * cblz * sblx) +
                      (calz * saly - caly * salx * salz) *
                          (cblz * sbly - cbly * sblx * sblz) +
                      calx * caly * cblx * cbly) -
                 cbcx * sbcy *
                     ((saly * salz + caly * calz * salx) *
                          (cbly * sblz - cblz * sblx * sbly) +
                      (calz * saly - caly * salx * salz) *
                          (cbly * cblz + sblx * sbly * sblz) -
                      calx * caly * cblx * sbly);
  transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]),
                                 crycrx / cos(transformTobeMapped[0]));

  float srzcrx =
      (cbcz * sbcy - cbcy * sbcx * sbcz) *
          (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
           calx * calz * (sbly * sblz + cbly * cblz * sblx) +
           cblx * cbly * salx) -
      (cbcy * cbcz + sbcx * sbcy * sbcz) *
          (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
           calx * salz * (cbly * cblz + sblx * sbly * sblz) +
           cblx * salx * sbly) +
      cbcx * sbcz *
          (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
  float crzcrx =
      (cbcy * sbcz - cbcz * sbcx * sbcy) *
          (calx * calz * (cbly * sblz - cblz * sblx * sbly) -
           calx * salz * (cbly * cblz + sblx * sbly * sblz) +
           cblx * salx * sbly) -
      (sbcy * sbcz + cbcy * cbcz * sbcx) *
          (calx * salz * (cblz * sbly - cbly * sblx * sblz) -
           calx * calz * (sbly * sblz + cbly * cblz * sblx) +
           cblx * cbly * salx) +
      cbcx * cbcz *
          (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
  transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]),
                                 crzcrx / cos(transformTobeMapped[0]));

  x1 = cos(transformTobeMapped[2]) * transformIncre[3] -
       sin(transformTobeMapped[2]) * transformIncre[4];
  y1 = sin(transformTobeMapped[2]) * transformIncre[3] +
       cos(transformTobeMapped[2]) * transformIncre[4];
  z1 = transformIncre[5];

  x2 = x1;
  y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
  z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

  transformTobeMapped[3] =
      transformAftMapped[3] -
      (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
  transformTobeMapped[4] = transformAftMapped[4] - y2;
  transformTobeMapped[5] =
      transformAftMapped[5] -
      (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
}

void MapOptimization::transformUpdate() {

  for (int i = 0; i < 6; i++) {
    transformBefMapped[i] = transformSum[i];
    transformAftMapped[i] = transformTobeMapped[i];
  }
}

void MapOptimization::updatePointAssociateToMapSinCos() {
  cRoll = cos(transformTobeMapped[0]);
  sRoll = sin(transformTobeMapped[0]);

  cPitch = cos(transformTobeMapped[1]);
  sPitch = sin(transformTobeMapped[1]);

  cYaw = cos(transformTobeMapped[2]);
  sYaw = sin(transformTobeMapped[2]);

  tX = transformTobeMapped[3];
  tY = transformTobeMapped[4];
  tZ = transformTobeMapped[5];
}

void MapOptimization::pointAssociateToMap(PointType const *const pi,
                                          PointType *const po) {
  float x1 = cYaw * pi->x - sYaw * pi->y;
  float y1 = sYaw * pi->x + cYaw * pi->y;
  float z1 = pi->z;

  float x2 = x1;
  float y2 = cRoll * y1 - sRoll * z1;
  float z2 = sRoll * y1 + cRoll * z1;

  po->x = cPitch * x2 + sPitch * z2 + tX;
  po->y = y2 + tY;
  po->z = -sPitch * x2 + cPitch * z2 + tZ;
  po->intensity = pi->intensity;
}

void MapOptimization::updateTransformPointCloudSinCos(PointTypePose *tIn) {
  ctRoll = cos(tIn->roll);
  stRoll = sin(tIn->roll);

  ctPitch = cos(tIn->pitch);
  stPitch = sin(tIn->pitch);

  ctYaw = cos(tIn->yaw);
  stYaw = sin(tIn->yaw);

  tInX = tIn->x;
  tInY = tIn->y;
  tInZ = tIn->z;
}

pcl::PointCloud<PointType>::Ptr MapOptimization::transformPointCloud(
    pcl::PointCloud<PointType>::Ptr cloudIn) {
  // !!! DO NOT use pcl for point cloud transformation, results are not
  // accurate Reason: unkown
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  PointType *pointFrom;
  PointType pointTo;

  int cloudSize = cloudIn->points.size();
  cloudOut->resize(cloudSize);

  for (int i = 0; i < cloudSize; ++i) {
    pointFrom = &cloudIn->points[i];
    float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
    float y1 = stYaw * pointFrom->x + ctYaw * pointFrom->y;
    float z1 = pointFrom->z;

    float x2 = x1;
    float y2 = ctRoll * y1 - stRoll * z1;
    float z2 = stRoll * y1 + ctRoll * z1;

    pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
    pointTo.y = y2 + tInY;
    pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
    pointTo.intensity = pointFrom->intensity;

    cloudOut->points[i] = pointTo;
  }
  return cloudOut;
}

pcl::PointCloud<PointType>::Ptr MapOptimization::transformPointCloud(
    pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn) {
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  PointType *pointFrom;
  PointType pointTo;

  int cloudSize = cloudIn->points.size();
  cloudOut->resize(cloudSize);

  for (int i = 0; i < cloudSize; ++i) {
    pointFrom = &cloudIn->points[i];
    float x1 = cos(transformIn->yaw) * pointFrom->x -
               sin(transformIn->yaw) * pointFrom->y;
    float y1 = sin(transformIn->yaw) * pointFrom->x +
               cos(transformIn->yaw) * pointFrom->y;
    float z1 = pointFrom->z;

    float x2 = x1;
    float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
    float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll) * z1;

    pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 +
                transformIn->x;
    pointTo.y = y2 + transformIn->y;
    pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 +
                transformIn->z;
    pointTo.intensity = pointFrom->intensity;

    cloudOut->points[i] = pointTo;
  }
  return cloudOut;
}


void MapOptimization::publishTF() {
  tf2::Quaternion q;
  geometry_msgs::msg::Quaternion geoQuat;
  q.setRPY(transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);
  // tf2::convert(q, geoQuat);
  // geoQuat = tf2::toMsg<tf2::Quaternion,geometry_msgs::msg::Quaternion>(q);
  geoQuat.x = q.x();
  geoQuat.y = q.y();
  geoQuat.z = q.z();
  geoQuat.w = q.w();

  odomAftMapped.header.stamp = timeLaserOdometry;
  odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
  odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
  odomAftMapped.pose.pose.orientation.z = geoQuat.x;
  odomAftMapped.pose.pose.orientation.w = geoQuat.w;
  odomAftMapped.pose.pose.position.x = transformAftMapped[3];
  odomAftMapped.pose.pose.position.y = transformAftMapped[4];
  odomAftMapped.pose.pose.position.z = transformAftMapped[5];
  odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
  odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
  odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
  odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
  odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
  odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
  pubOdomAftMapped->publish(odomAftMapped);

  aftMappedTrans.header.stamp = timeLaserOdometry;
  aftMappedTrans.transform.translation.x = transformAftMapped[3];
  aftMappedTrans.transform.translation.y = transformAftMapped[4];
  aftMappedTrans.transform.translation.z = transformAftMapped[5];
  aftMappedTrans.transform.rotation.x = -geoQuat.y;
  aftMappedTrans.transform.rotation.y = -geoQuat.z;
  aftMappedTrans.transform.rotation.z = geoQuat.x;
  aftMappedTrans.transform.rotation.w = geoQuat.w;
  tfBroadcaster->sendTransform(aftMappedTrans);
}

void MapOptimization::publishKeyPosesAndFrames() {
  if (pubKeyPoses->get_subscription_count() != 0) {
    sensor_msgs::msg::PointCloud2 cloudMsgTemp;
    pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
    cloudMsgTemp.header.stamp = timeLaserOdometry;
    cloudMsgTemp.header.frame_id = "camera_init";
    pubKeyPoses->publish(cloudMsgTemp);
  }

  if (pubRecentKeyFrames->get_subscription_count() != 0) {
    sensor_msgs::msg::PointCloud2 cloudMsgTemp;
    pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
    cloudMsgTemp.header.stamp = timeLaserOdometry;
    cloudMsgTemp.header.frame_id = "camera_init";
    pubRecentKeyFrames->publish(cloudMsgTemp);
  }
}

void MapOptimization::publishGlobalMap() {
  if (pubLaserCloudSurround->get_subscription_count() == 0) return;

  if (cloudKeyPoses3D->points.empty() == true) return;
  // kd-tree to find near key frames to visualize
  std::vector<int> pointSearchIndGlobalMap;
  std::vector<float> pointSearchSqDisGlobalMap;
  // search near key frames to visualize
  mtx.lock();
  kdtreeGlobalMap.setInputCloud(cloudKeyPoses3D);
  kdtreeGlobalMap.radiusSearch(
      currentRobotPosPoint, _global_map_visualization_search_radius,
      pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
  mtx.unlock();

  for (size_t i = 0; i < pointSearchIndGlobalMap.size(); ++i)
    globalMapKeyPoses->points.push_back(
        cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
  // downsample near selected key frames
  downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
  downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
  // extract visualized and downsampled key frames
  for (size_t i = 0; i < globalMapKeyPosesDS->points.size(); ++i) {
    int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
    *globalMapKeyFrames += *transformPointCloud(
        cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    *globalMapKeyFrames += *transformPointCloud(
        surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    *globalMapKeyFrames +=
        *transformPointCloud(outlierCloudKeyFrames[thisKeyInd],
                             &cloudKeyPoses6D->points[thisKeyInd]);
  }
  // downsample visualized points
  downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
  downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

  sensor_msgs::msg::PointCloud2 cloudMsgTemp;
  pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
  cloudMsgTemp.header.stamp = timeLaserOdometry;
  cloudMsgTemp.header.frame_id = "camera_init";
  pubLaserCloudSurround->publish(cloudMsgTemp);

  globalMapKeyPoses->clear();
  globalMapKeyPosesDS->clear();
  globalMapKeyFrames->clear();
  //globalMapKeyFramesDS->clear();
}

bool MapOptimization::detectLoopClosure() {
  latestSurfKeyFrameCloud->clear();
  nearHistorySurfKeyFrameCloud->clear();
  nearHistorySurfKeyFrameCloudDS->clear();

  std::lock_guard<std::mutex> lock(mtx);
  // find the closest history key frame
  std::vector<int> pointSearchIndLoop;
  std::vector<float> pointSearchSqDisLoop;
  kdtreeHistoryKeyPoses.setInputCloud(cloudKeyPoses3D);
  kdtreeHistoryKeyPoses.radiusSearch(
      currentRobotPosPoint, _history_keyframe_search_radius, pointSearchIndLoop,
      pointSearchSqDisLoop);

  closestHistoryFrameID = -1;
  for (size_t i = 0; i < pointSearchIndLoop.size(); ++i) {
    int id = pointSearchIndLoop[i];
    if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry.seconds()) > 30.0) {
      closestHistoryFrameID = id;
      break;
    }
  }
  if (closestHistoryFrameID == -1) {
    return false;
  }
  // save latest key frames
  latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
  *latestSurfKeyFrameCloud +=
      *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure],
                           &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
  *latestSurfKeyFrameCloud +=
      *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],
                           &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

  pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
  int cloudSize = latestSurfKeyFrameCloud->points.size();
  for (int i = 0; i < cloudSize; ++i) {
    if ((int)latestSurfKeyFrameCloud->points[i].intensity >= 0) {
      hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);
    }
  }
  latestSurfKeyFrameCloud->clear();
  *latestSurfKeyFrameCloud = *hahaCloud;
  // save history near key frames
  for (int j = - _history_keyframe_search_num; j <= _history_keyframe_search_num; ++j) {
    if (closestHistoryFrameID + j < 0 ||
        closestHistoryFrameID + j > latestFrameIDLoopCloure)
      continue;
    *nearHistorySurfKeyFrameCloud += *transformPointCloud(
        cornerCloudKeyFrames[closestHistoryFrameID + j],
        &cloudKeyPoses6D->points[closestHistoryFrameID + j]);
    *nearHistorySurfKeyFrameCloud += *transformPointCloud(
        surfCloudKeyFrames[closestHistoryFrameID + j],
        &cloudKeyPoses6D->points[closestHistoryFrameID + j]);
  }

  downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);
  downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS);
  // publish history near key frames
  if (pubHistoryKeyFrames->get_subscription_count() != 0) {
    sensor_msgs::msg::PointCloud2 cloudMsgTemp;
    pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
    cloudMsgTemp.header.stamp = timeLaserOdometry;
    cloudMsgTemp.header.frame_id = "camera_init";
    pubHistoryKeyFrames->publish(cloudMsgTemp);
  }

  return true;
}

void MapOptimization::performLoopClosure() {

  if (cloudKeyPoses3D->points.empty() == true)
    return;


  // try to find close key frame if there are any
  if (potentialLoopFlag == false) {
    if (detectLoopClosure() == true) {
      potentialLoopFlag = true;  // find some key frames that is old enough or
                                 // close enough for loop closure
    }
    if (potentialLoopFlag == false) return;
  }
  // reset the flag first no matter icp successes or not
  potentialLoopFlag = false;
  // ICP Settings
  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setMaxCorrespondenceDistance(100);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);
  // Align clouds
  icp.setInputSource(latestSurfKeyFrameCloud);
  icp.setInputTarget(nearHistorySurfKeyFrameCloudDS);
  pcl::PointCloud<PointType>::Ptr unused_result(
      new pcl::PointCloud<PointType>());
  icp.align(*unused_result);

  if (icp.hasConverged() == false ||
      icp.getFitnessScore() > _history_keyframe_fitness_score)
    return;
  // publish corrected cloud
  if (pubIcpKeyFrames->get_subscription_count() != 0) {
    pcl::PointCloud<PointType>::Ptr closed_cloud(
        new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*latestSurfKeyFrameCloud, *closed_cloud,
                             icp.getFinalTransformation());
    sensor_msgs::msg::PointCloud2 cloudMsgTemp;
    pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
    cloudMsgTemp.header.stamp = timeLaserOdometry;
    cloudMsgTemp.header.frame_id = "camera_init";
    pubIcpKeyFrames->publish(cloudMsgTemp);
  }
  /*
          get pose constraint
          */
  float x, y, z, roll, pitch, yaw;
  Eigen::Affine3f correctionCameraFrame;
  correctionCameraFrame =
      icp.getFinalTransformation();  // get transformation in camera frame
                                     // (because points are in camera frame)
  pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch,
                                    yaw);
  Eigen::Affine3f correctionLidarFrame =
      pcl::getTransformation(z, x, y, yaw, roll, pitch);
  // transform from world origin to wrong pose
  Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(
      cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
  // transform from world origin to corrected pose
  Eigen::Affine3f tCorrect =
      correctionLidarFrame *
      tWrong;  // pre-multiplying -> successive rotation about a fixed frame
  pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
  gtsam::Pose3 poseFrom =
      Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
  gtsam::Pose3 poseTo =
      pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);
  gtsam::Vector Vector6(6);
  float noiseScore = icp.getFitnessScore();
  Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
      noiseScore;
  auto constraintNoise = noiseModel::Diagonal::Variances(Vector6);
  /*
          add constraints
          */
  std::lock_guard<std::mutex> lock(mtx);
  gtSAMgraph.add(
      BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID,
                           poseFrom.between(poseTo), constraintNoise));
  isam->update(gtSAMgraph);
  isam->update();
  gtSAMgraph.resize(0);

  aLoopIsClosed = true;
}

void MapOptimization::extractSurroundingKeyFrames() {
  if (cloudKeyPoses3D->points.empty() == true) return;

  if (_loop_closure_enabled == true) {
    // only use recent key poses for graph building
    if (static_cast<int>(recentCornerCloudKeyFrames.size()) <
        _surrounding_keyframe_search_num) {  // queue is not full (the beginning
                                         // of mapping or a loop is just
                                         // closed)
                                         // clear recent key frames queue
      recentCornerCloudKeyFrames.clear();
      recentSurfCloudKeyFrames.clear();
      recentOutlierCloudKeyFrames.clear();
      int numPoses = cloudKeyPoses3D->points.size();
      for (int i = numPoses - 1; i >= 0; --i) {
        int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
        PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
        updateTransformPointCloudSinCos(&thisTransformation);
        // extract surrounding map
        recentCornerCloudKeyFrames.push_front(
            transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
        recentSurfCloudKeyFrames.push_front(
            transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
        recentOutlierCloudKeyFrames.push_front(
            transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
        if (static_cast<int>(recentCornerCloudKeyFrames.size()) >= _surrounding_keyframe_search_num)
          break;
      }
    } else {  // queue is full, pop the oldest key frame and push the latest
              // key frame
      if (latestFrameID != static_cast<int>(cloudKeyPoses3D->points.size()) - 1) {
        // if the robot is not moving, no need to
        // update recent frames

        recentCornerCloudKeyFrames.pop_front();
        recentSurfCloudKeyFrames.pop_front();
        recentOutlierCloudKeyFrames.pop_front();
        // push latest scan to the end of queue
        latestFrameID = cloudKeyPoses3D->points.size() - 1;
        PointTypePose thisTransformation =
            cloudKeyPoses6D->points[latestFrameID];
        updateTransformPointCloudSinCos(&thisTransformation);
        recentCornerCloudKeyFrames.push_back(
            transformPointCloud(cornerCloudKeyFrames[latestFrameID]));
        recentSurfCloudKeyFrames.push_back(
            transformPointCloud(surfCloudKeyFrames[latestFrameID]));
        recentOutlierCloudKeyFrames.push_back(
            transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
      }
    }

    for (size_t i = 0; i < recentCornerCloudKeyFrames.size(); ++i) {
      *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
      *laserCloudSurfFromMap += *recentSurfCloudKeyFrames[i];
      *laserCloudSurfFromMap += *recentOutlierCloudKeyFrames[i];
    }
  } else {
    surroundingKeyPoses->clear();
    surroundingKeyPosesDS->clear();
    // extract all the nearby key poses and downsample them
    kdtreeSurroundingKeyPoses.setInputCloud(cloudKeyPoses3D);
    kdtreeSurroundingKeyPoses.radiusSearch(
        currentRobotPosPoint, (double)_surrounding_keyframe_search_radius,
        pointSearchInd, pointSearchSqDis);

    for (size_t i = 0; i < pointSearchInd.size(); ++i){
      surroundingKeyPoses->points.push_back(
          cloudKeyPoses3D->points[pointSearchInd[i]]);
    }

    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

    // delete key frames that are not in surrounding region
    int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
    for (size_t i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
      bool existingFlag = false;
      for (int j = 0; j < numSurroundingPosesDS; ++j) {
        if (surroundingExistingKeyPosesID[i] ==
            (int)surroundingKeyPosesDS->points[j].intensity) {
          existingFlag = true;
          break;
        }
      }
      if (existingFlag == false) {
        surroundingExistingKeyPosesID.erase(
            surroundingExistingKeyPosesID.begin() + i);
        surroundingCornerCloudKeyFrames.erase(
            surroundingCornerCloudKeyFrames.begin() + i);
        surroundingSurfCloudKeyFrames.erase(
            surroundingSurfCloudKeyFrames.begin() + i);
        surroundingOutlierCloudKeyFrames.erase(
            surroundingOutlierCloudKeyFrames.begin() + i);
        --i;
      }
    }
    // add new key frames that are not in calculated existing key frames
    for (int i = 0; i < numSurroundingPosesDS; ++i) {
      bool existingFlag = false;
      for (auto iter = surroundingExistingKeyPosesID.begin();
           iter != surroundingExistingKeyPosesID.end(); ++iter) {
        if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity) {
          existingFlag = true;
          break;
        }
      }
      if (existingFlag == true) {
        continue;
      } else {
        int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
        PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
        updateTransformPointCloudSinCos(&thisTransformation);
        surroundingExistingKeyPosesID.push_back(thisKeyInd);
        surroundingCornerCloudKeyFrames.push_back(
            transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
        surroundingSurfCloudKeyFrames.push_back(
            transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
        surroundingOutlierCloudKeyFrames.push_back(
            transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
      }
    }

    for (size_t i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
      *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
      *laserCloudSurfFromMap += *surroundingSurfCloudKeyFrames[i];
      *laserCloudSurfFromMap += *surroundingOutlierCloudKeyFrames[i];
    }
  }
  // Downsample the surrounding corner key frames (or map)
  downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
  downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
  laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();
  // Downsample the surrounding surf key frames (or map)
  downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
  downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
  laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
}

void MapOptimization::downsampleCurrentScan() {
  laserCloudCornerLastDS->clear();
  downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
  downSizeFilterCorner.filter(*laserCloudCornerLastDS);
  laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();

  laserCloudSurfLastDS->clear();
  downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
  downSizeFilterSurf.filter(*laserCloudSurfLastDS);
  laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();

  laserCloudOutlierLastDS->clear();
  downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);
  downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
  laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size();

  laserCloudSurfTotalLast->clear();
  laserCloudSurfTotalLastDS->clear();
  *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
  *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
  downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
  downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
  laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
}

void MapOptimization::cornerOptimization() {
  updatePointAssociateToMapSinCos();
  for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
    pointOri = laserCloudCornerLastDS->points[i];
    pointAssociateToMap(&pointOri, &pointSel);
    kdtreeCornerFromMap.nearestKSearch(pointSel, 5, pointSearchInd,
                                       pointSearchSqDis);

    if (pointSearchSqDis[4] < 1.0) {
      float cx = 0, cy = 0, cz = 0;
      for (int j = 0; j < 5; j++) {
        cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
        cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
        cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
      }
      cx /= 5;
      cy /= 5;
      cz /= 5;

      float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
      for (int j = 0; j < 5; j++) {
        float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
        float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
        float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

        a11 += ax * ax;
        a12 += ax * ay;
        a13 += ax * az;
        a22 += ay * ay;
        a23 += ay * az;
        a33 += az * az;
      }
      a11 /= 5;
      a12 /= 5;
      a13 /= 5;
      a22 /= 5;
      a23 /= 5;
      a33 /= 5;

      matA1(0, 0) = a11;
      matA1(0, 1) = a12;
      matA1(0, 2) = a13;
      matA1(1, 0) = a12;
      matA1(1, 1) = a22;
      matA1(1, 2) = a23;
      matA1(2, 0) = a13;
      matA1(2, 1) = a23;
      matA1(2, 2) = a33;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> esolver(matA1);

      matD1 = esolver.eigenvalues().real();
      matV1 = esolver.eigenvectors().real();

      if (matD1[2] > 3 * matD1[1]) {
        float x0 = pointSel.x;
        float y0 = pointSel.y;
        float z0 = pointSel.z;
        float x1 = cx + 0.1 * matV1(0, 0);
        float y1 = cy + 0.1 * matV1(0, 1);
        float z1 = cz + 0.1 * matV1(0, 2);
        float x2 = cx - 0.1 * matV1(0, 0);
        float y2 = cy - 0.1 * matV1(0, 1);
        float z2 = cz - 0.1 * matV1(0, 2);

        float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                              ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                          ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                              ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                          ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                              ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

        float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                         (z1 - z2) * (z1 - z2));

        float la =
            ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
             (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
            a012 / l12;

        float lb =
            -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
              (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
            a012 / l12;

        float lc =
            -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
              (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
            a012 / l12;

        float ld2 = a012 / l12;

        float s = 1 - 0.9 * fabs(ld2);

        coeff.x = s * la;
        coeff.y = s * lb;
        coeff.z = s * lc;
        coeff.intensity = s * ld2;

        if (s > 0.1) {
          laserCloudOri->push_back(pointOri);
          coeffSel->push_back(coeff);
        }
      }
    }
  }
}

void MapOptimization::surfOptimization() {
  updatePointAssociateToMapSinCos();
  for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++) {
    pointOri = laserCloudSurfTotalLastDS->points[i];
    pointAssociateToMap(&pointOri, &pointSel);
    kdtreeSurfFromMap.nearestKSearch(pointSel, 5, pointSearchInd,
                                      pointSearchSqDis);

    if (pointSearchSqDis[4] < 1.0) {
      for (int j = 0; j < 5; j++) {
        matA0(j, 0) =
            laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
        matA0(j, 1) =
            laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
        matA0(j, 2) =
            laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
      }
      matX0 = matA0.colPivHouseholderQr().solve(matB0);

      float pa = matX0(0, 0);
      float pb = matX0(1, 0);
      float pc = matX0(2, 0);
      float pd = 1;

      float ps = sqrt(pa * pa + pb * pb + pc * pc);
      pa /= ps;
      pb /= ps;
      pc /= ps;
      pd /= ps;

      bool planeValid = true;
      for (int j = 0; j < 5; j++) {
        if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                 pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                 pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z +
                 pd) > 0.2) {
          planeValid = false;
          break;
        }
      }

      if (planeValid) {
        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

        float s = 1 - 0.9 * fabs(pd2) /
                          sqrt(sqrt(pointSel.x * pointSel.x +
                                    pointSel.y * pointSel.y +
                                    pointSel.z * pointSel.z));

        coeff.x = s * pa;
        coeff.y = s * pb;
        coeff.z = s * pc;
        coeff.intensity = s * pd2;

        if (s > 0.1) {
          laserCloudOri->push_back(pointOri);
          coeffSel->push_back(coeff);
        }
      }
    }
  }
}

bool MapOptimization::LMOptimization(int iterCount) {
  float srx = sin(transformTobeMapped[0]);
  float crx = cos(transformTobeMapped[0]);
  float sry = sin(transformTobeMapped[1]);
  float cry = cos(transformTobeMapped[1]);
  float srz = sin(transformTobeMapped[2]);
  float crz = cos(transformTobeMapped[2]);

  int laserCloudSelNum = laserCloudOri->points.size();
  if (laserCloudSelNum < 50) {
    return false;
  }

  Eigen::Matrix<float,Eigen::Dynamic,6> matA(laserCloudSelNum, 6);
  Eigen::Matrix<float,6,Eigen::Dynamic> matAt(6,laserCloudSelNum);
  Eigen::Matrix<float,6,6> matAtA;
  Eigen::VectorXf matB(laserCloudSelNum);
  Eigen::Matrix<float,6,1> matAtB;
  Eigen::Matrix<float,6,1> matX;

  for (int i = 0; i < laserCloudSelNum; i++) {
    pointOri = laserCloudOri->points[i];
    coeff = coeffSel->points[i];

    float arx =
        (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y -
         srx * sry * pointOri.z) *
            coeff.x +
        (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) *
            coeff.y +
        (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y -
         cry * srx * pointOri.z) *
            coeff.z;

    float ary =
        ((cry * srx * srz - crz * sry) * pointOri.x +
         (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) *
            coeff.x +
        ((-cry * crz - srx * sry * srz) * pointOri.x +
         (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) *
            coeff.z;

    float arz = ((crz * srx * sry - cry * srz) * pointOri.x +
                 (-cry * crz - srx * sry * srz) * pointOri.y) *
                    coeff.x +
                (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                ((sry * srz + cry * crz * srx) * pointOri.x +
                 (crz * sry - cry * srx * srz) * pointOri.y) *
                    coeff.z;

    matA(i, 0) = arx;
    matA(i, 1) = ary;
    matA(i, 2) = arz;
    matA(i, 3) = coeff.x;
    matA(i, 4) = coeff.y;
    matA(i, 5) = coeff.z;
    matB(i, 0) = -coeff.intensity;
  }
  matAt = matA.transpose();
  matAtA = matAt * matA;
  matAtB = matAt * matB;
  matX = matAtA.colPivHouseholderQr().solve(matAtB);

  if (iterCount == 0) {
    Eigen::Matrix<float,1,6> matE;
    Eigen::Matrix<float,6,6> matV;
    Eigen::Matrix<float,6,6> matV2;

    Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float,6, 6> > esolver(matAtA);
    matE = esolver.eigenvalues().real();
    matV = esolver.eigenvectors().real();

     matV2 = matV;

    isDegenerate = false;
    float eignThre[6] = {100, 100, 100, 100, 100, 100};
    for (int i = 5; i >= 0; i--) {
      if (matE(0, i) < eignThre[i]) {
        for (int j = 0; j < 6; j++) {
          matV2(i, j) = 0;
        }
        isDegenerate = true;
      } else {
        break;
      }
    }
    matP = matV.inverse() * matV2;
  }

  if (isDegenerate) {
    Eigen::Matrix<float,6, 1> matX2(matX);
    matX2 = matX;
    matX = matP * matX2;
  }

  transformTobeMapped[0] += matX(0, 0);
  transformTobeMapped[1] += matX(1, 0);
  transformTobeMapped[2] += matX(2, 0);
  transformTobeMapped[3] += matX(3, 0);
  transformTobeMapped[4] += matX(4, 0);
  transformTobeMapped[5] += matX(5, 0);

  float deltaR = sqrt(pow(pcl::rad2deg(matX(0, 0)), 2) +
                      pow(pcl::rad2deg(matX(1, 0)), 2) +
                      pow(pcl::rad2deg(matX(2, 0)), 2));
  float deltaT = sqrt(pow(matX(3, 0) * 100, 2) +
                      pow(matX(4, 0) * 100, 2) +
                      pow(matX(5, 0) * 100, 2));

  if (deltaR < 0.05 && deltaT < 0.05) {
    return true;
  }
  return false;
}

void MapOptimization::scan2MapOptimization() {
  if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) {
    kdtreeCornerFromMap.setInputCloud(laserCloudCornerFromMapDS);
    kdtreeSurfFromMap.setInputCloud(laserCloudSurfFromMapDS);

    for (int iterCount = 0; iterCount < 10; iterCount++) {
      laserCloudOri->clear();
      coeffSel->clear();

      cornerOptimization();
      surfOptimization();

      if (LMOptimization(iterCount) == true) break;
    }

    transformUpdate();
  }
}

void MapOptimization::saveKeyFramesAndFactor() {
  currentRobotPosPoint.x = transformAftMapped[3];
  currentRobotPosPoint.y = transformAftMapped[4];
  currentRobotPosPoint.z = transformAftMapped[5];

  gtsam::Vector Vector6(6);
  Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
  auto priorNoise = noiseModel::Diagonal::Variances(Vector6);
  auto odometryNoise = noiseModel::Diagonal::Variances(Vector6);

  bool saveThisKeyFrame = true;
  if (sqrt((previousRobotPosPoint.x - currentRobotPosPoint.x) *
               (previousRobotPosPoint.x - currentRobotPosPoint.x) +
           (previousRobotPosPoint.y - currentRobotPosPoint.y) *
               (previousRobotPosPoint.y - currentRobotPosPoint.y) +
           (previousRobotPosPoint.z - currentRobotPosPoint.z) *
               (previousRobotPosPoint.z - currentRobotPosPoint.z)) < 0.3) {
    saveThisKeyFrame = false;
  }

  if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty()) return;

  previousRobotPosPoint = currentRobotPosPoint;
  /**
   * update grsam graph
   */
  if (cloudKeyPoses3D->points.empty()) {
    gtSAMgraph.add(PriorFactor<Pose3>(
        0,
        Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0],
                           transformTobeMapped[1]),
              Point3(transformTobeMapped[5], transformTobeMapped[3],
                     transformTobeMapped[4])),
        priorNoise));
    initialEstimate.insert(
        0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0],
                              transformTobeMapped[1]),
                 Point3(transformTobeMapped[5], transformTobeMapped[3],
                        transformTobeMapped[4])));
    for (int i = 0; i < 6; ++i) transformLast[i] = transformTobeMapped[i];
  } else {
    gtsam::Pose3 poseFrom = Pose3(
        Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
        Point3(transformLast[5], transformLast[3], transformLast[4]));
    gtsam::Pose3 poseTo =
        Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0],
                           transformAftMapped[1]),
              Point3(transformAftMapped[5], transformAftMapped[3],
                     transformAftMapped[4]));
    gtSAMgraph.add(BetweenFactor<Pose3>(
        cloudKeyPoses3D->points.size() - 1, cloudKeyPoses3D->points.size(),
        poseFrom.between(poseTo), odometryNoise));
    initialEstimate.insert(
        cloudKeyPoses3D->points.size(),
        Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0],
                           transformAftMapped[1]),
              Point3(transformAftMapped[5], transformAftMapped[3],
                     transformAftMapped[4])));
  }
  /**
   * update iSAM
   */
  isam->update(gtSAMgraph, initialEstimate);
  isam->update();

  gtSAMgraph.resize(0);
  initialEstimate.clear();

  /**
   * save key poses
   */
  PointType thisPose3D;
  PointTypePose thisPose6D;
  Pose3 latestEstimate;

  isamCurrentEstimate = isam->calculateEstimate();
  latestEstimate =
      isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);

  thisPose3D.x = latestEstimate.translation().y();
  thisPose3D.y = latestEstimate.translation().z();
  thisPose3D.z = latestEstimate.translation().x();
  thisPose3D.intensity =
      cloudKeyPoses3D->points.size();  // this can be used as index
  cloudKeyPoses3D->push_back(thisPose3D);

  thisPose6D.x = thisPose3D.x;
  thisPose6D.y = thisPose3D.y;
  thisPose6D.z = thisPose3D.z;
  thisPose6D.intensity = thisPose3D.intensity;  // this can be used as index
  thisPose6D.roll = latestEstimate.rotation().pitch();
  thisPose6D.pitch = latestEstimate.rotation().yaw();
  thisPose6D.yaw = latestEstimate.rotation().roll();  // in camera frame
  thisPose6D.time = timeLaserOdometry.seconds();
  cloudKeyPoses6D->push_back(thisPose6D);
  /**
   * save updated transform
   */
  if (cloudKeyPoses3D->points.size() > 1) {
    transformAftMapped[0] = latestEstimate.rotation().pitch();
    transformAftMapped[1] = latestEstimate.rotation().yaw();
    transformAftMapped[2] = latestEstimate.rotation().roll();
    transformAftMapped[3] = latestEstimate.translation().y();
    transformAftMapped[4] = latestEstimate.translation().z();
    transformAftMapped[5] = latestEstimate.translation().x();

    for (int i = 0; i < 6; ++i) {
      transformLast[i] = transformAftMapped[i];
      transformTobeMapped[i] = transformAftMapped[i];
    }
  }

  pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(
      new pcl::PointCloud<PointType>());

  pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
  pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
  pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);

  cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
  surfCloudKeyFrames.push_back(thisSurfKeyFrame);
  outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);
}

void MapOptimization::correctPoses() {
  if (aLoopIsClosed == true) {
    recentCornerCloudKeyFrames.clear();
    recentSurfCloudKeyFrames.clear();
    recentOutlierCloudKeyFrames.clear();
    // update key poses
    int numPoses = isamCurrentEstimate.size();
    for (int i = 0; i < numPoses; ++i) {
      cloudKeyPoses3D->points[i].x =
          isamCurrentEstimate.at<Pose3>(i).translation().y();
      cloudKeyPoses3D->points[i].y =
          isamCurrentEstimate.at<Pose3>(i).translation().z();
      cloudKeyPoses3D->points[i].z =
          isamCurrentEstimate.at<Pose3>(i).translation().x();

      cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
      cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
      cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
      cloudKeyPoses6D->points[i].roll =
          isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
      cloudKeyPoses6D->points[i].pitch =
          isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
      cloudKeyPoses6D->points[i].yaw =
          isamCurrentEstimate.at<Pose3>(i).rotation().roll();
    }

    aLoopIsClosed = false;
  }
}

void MapOptimization::clearCloud() {
  laserCloudCornerFromMap->clear();
  laserCloudSurfFromMap->clear();
  laserCloudCornerFromMapDS->clear();
  laserCloudSurfFromMapDS->clear();
}


void MapOptimization::run() {
  size_t cycle_count = 0;

  while (rclcpp::ok()) {
    AssociationOut association;
    _input_channel.receive(association);
    if( !rclcpp::ok() ) break;

    {
      std::lock_guard<std::mutex> lock(mtx);

      laserCloudCornerLast = association.cloud_corner_last;
      laserCloudSurfLast = association.cloud_surf_last;
      laserCloudOutlierLast = association.cloud_outlier_last;

      timeLaserOdometry = association.laser_odometry.header.stamp;

      OdometryToTransform(association.laser_odometry, transformSum);

      transformAssociateToMap();

      extractSurroundingKeyFrames();

      downsampleCurrentScan();

      scan2MapOptimization();

      saveKeyFramesAndFactor();

      correctPoses();

      publishTF();

      publishKeyPosesAndFrames();

      clearCloud();
    }
    cycle_count++;

    if ((cycle_count % 3) == 0) {
      _loop_closure_signal.send(true);
    }

    if ((cycle_count % 10) == 0) {
      _publish_global_signal.send(true);
    }
  }
}
