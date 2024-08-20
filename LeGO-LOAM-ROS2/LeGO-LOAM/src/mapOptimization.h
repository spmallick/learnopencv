#ifndef MAPOPTIMIZATION_H
#define MAPOPTIMIZATION_H

#include "lego_loam/utility.h"
#include "lego_loam/channel.h"
#include "lego_loam/nanoflann_pcl.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ISAM2.h>

inline gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
  // camera frame to lidar frame
  return gtsam::Pose3(
      gtsam::Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll),
                          double(thisPoint.pitch)),
      gtsam::Point3(double(thisPoint.z), double(thisPoint.x),
                    double(thisPoint.y)));
}

inline Eigen::Affine3f pclPointToAffine3fCameraToLidar(
    PointTypePose thisPoint) {
  // camera frame to lidar frame
  return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y,
                                thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
}


class MapOptimization : public rclcpp::Node {

 public:
  MapOptimization(const std::string &name, Channel<AssociationOut> &input_channel);

  ~MapOptimization();

  void run();

 private:
  gtsam::NonlinearFactorGraph gtSAMgraph;
  gtsam::Values initialEstimate;
  gtsam::ISAM2 *isam;
  gtsam::Values isamCurrentEstimate;

  bool _loop_closure_enabled;

  float _surrounding_keyframe_search_radius;
  int   _surrounding_keyframe_search_num;
  float _history_keyframe_search_radius;
  int   _history_keyframe_search_num;
  float _history_keyframe_fitness_score;
  float _global_map_visualization_search_radius;

  Channel<AssociationOut>& _input_channel;
  std::thread _run_thread;

  Channel<bool> _publish_global_signal;
  std::thread _publish_global_thread;
  void publishGlobalMapThread();

  Channel<bool> _loop_closure_signal;
  std::thread _loop_closure_thread;
  void loopClosureThread();

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;

  nav_msgs::msg::Odometry odomAftMapped;
  geometry_msgs::msg::TransformStamped aftMappedTrans;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;

  std::vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
  std::vector<pcl::PointCloud<PointType>::Ptr> outlierCloudKeyFrames;

  std::deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;
  std::deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
  std::deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames;
  int latestFrameID;

  std::vector<int> surroundingExistingKeyPosesID;
  std::deque<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyFrames;
  std::deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;
  std::deque<pcl::PointCloud<PointType>::Ptr> surroundingOutlierCloudKeyFrames;

  PointType previousRobotPosPoint;
  PointType currentRobotPosPoint;

  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

  pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;
  pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

  pcl::PointCloud<PointType>::Ptr
      laserCloudCornerLast;  // corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr
      laserCloudSurfLast;  // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr
      laserCloudCornerLastDS;  // downsampled corner featuer set from
      // odoOptimization
  pcl::PointCloud<PointType>::Ptr
      laserCloudSurfLastDS;  // downsampled surf featuer set from
      // odoOptimization

  pcl::PointCloud<PointType>::Ptr
      laserCloudOutlierLast;  // corner feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr
      laserCloudOutlierLastDS;  // corner feature set from odoOptimization

  pcl::PointCloud<PointType>::Ptr
      laserCloudSurfTotalLast;  // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr
      laserCloudSurfTotalLastDS;  // downsampled corner featuer set from
      // odoOptimization

  pcl::PointCloud<PointType>::Ptr laserCloudOri;
  pcl::PointCloud<PointType>::Ptr coeffSel;

  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

  nanoflann::KdTreeFLANN<PointType> kdtreeCornerFromMap;
  nanoflann::KdTreeFLANN<PointType> kdtreeSurfFromMap;

  nanoflann::KdTreeFLANN<PointType> kdtreeSurroundingKeyPoses;
  nanoflann::KdTreeFLANN<PointType> kdtreeHistoryKeyPoses;

  pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;
  pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloudDS;

  pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloud;
  pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;

  nanoflann::KdTreeFLANN<PointType> kdtreeGlobalMap;
  pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
  pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
  pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
  pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;

  std::vector<int> pointSearchInd;
  std::vector<float> pointSearchSqDis;

  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterOutlier;
  pcl::VoxelGrid<PointType>
      downSizeFilterHistoryKeyFrames;  // for histor key frames of loop closure
  pcl::VoxelGrid<PointType>
      downSizeFilterSurroundingKeyPoses;  // for surrounding key poses of
      // scan-to-map optimization
  pcl::VoxelGrid<PointType>
      downSizeFilterGlobalMapKeyPoses;  // for global map visualization
  pcl::VoxelGrid<PointType>
      downSizeFilterGlobalMapKeyFrames;  // for global map visualization

  rclcpp::Time timeLaserOdometry;

  float transformLast[6];
  float transformSum[6];
  float transformIncre[6];
  float transformTobeMapped[6];
  float transformBefMapped[6];
  float transformAftMapped[6];

  std::mutex mtx;

  PointType pointOri, pointSel, pointProj, coeff;

  Eigen::Matrix<float, 5, 3> matA0;
  Eigen::Matrix<float, 5, 1> matB0;
  Eigen::Vector3f matX0;

  Eigen::Matrix3f matA1;
  Eigen::Matrix<float, 1, 3> matD1;
  Eigen::Matrix3f matV1;

  Eigen::Matrix<float, 6, 6> matP;

  bool isDegenerate;

  int laserCloudCornerFromMapDSNum;
  int laserCloudSurfFromMapDSNum;
  int laserCloudCornerLastDSNum;
  int laserCloudSurfLastDSNum;
  int laserCloudOutlierLastDSNum;
  int laserCloudSurfTotalLastDSNum;

  bool potentialLoopFlag;
  int closestHistoryFrameID;
  int latestFrameIDLoopCloure;

  bool aLoopIsClosed;

  float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
  float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

 private:
  void allocateMemory();
  void transformAssociateToMap();
  void transformUpdate();
  void updatePointAssociateToMapSinCos();
  void pointAssociateToMap(PointType const *const pi, PointType *const po);
  void updateTransformPointCloudSinCos(PointTypePose *tIn) ;

  pcl::PointCloud<PointType>::Ptr transformPointCloud(
      pcl::PointCloud<PointType>::Ptr cloudIn);

  pcl::PointCloud<PointType>::Ptr transformPointCloud(
      pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn);

  void publishTF();
  void publishKeyPosesAndFrames();
  void publishGlobalMap();

  bool detectLoopClosure();
  void performLoopClosure();

  void extractSurroundingKeyFrames();
  void downsampleCurrentScan();
  void cornerOptimization();
  void surfOptimization();

  bool LMOptimization(int iterCount);
  void scan2MapOptimization();

  void saveKeyFramesAndFactor();
  void correctPoses();

  void clearCloud();
};

#endif // MAPOPTIMIZATION_H
