cmake_minimum_required(VERSION 3.5)
project(pcl_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
endif()

## Find system dependencies
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED QUIET COMPONENTS common features filters io segmentation surface)

## Find ROS package dependencies
find_package(ament_cmake REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

set(dependencies
  pcl_conversions
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  tf2
  tf2_eigen
  tf2_geometry_msgs
  tf2_ros
  EIGEN3
  PCL
)

## Declare the pcl_ros_tf library
add_library(pcl_ros_tf src/transforms.cpp)
target_include_directories(pcl_ros_tf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
ament_target_dependencies(pcl_ros_tf
  ${dependencies}
)

### Nodelets
#
### Declare the pcl_ros_io library
#add_library(pcl_ros_io
#  src/pcl_ros/io/bag_io.cpp
#  src/pcl_ros/io/concatenate_data.cpp
#  src/pcl_ros/io/concatenate_fields.cpp
#  src/pcl_ros/io/io.cpp
#  src/pcl_ros/io/pcd_io.cpp
#)
#target_link_libraries(pcl_ros_io pcl_ros_tf ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#class_loader_hide_library_symbols(pcl_ros_io)
#
### Declare the pcl_ros_features library
#add_library(pcl_ros_features
#  src/pcl_ros/features/feature.cpp
#  # Compilation is much faster if we include all the following CPP files in feature.cpp
#  src/pcl_ros/features/boundary.cpp
#  src/pcl_ros/features/fpfh.cpp
#  src/pcl_ros/features/fpfh_omp.cpp
#  src/pcl_ros/features/shot.cpp
#  src/pcl_ros/features/shot_omp.cpp
#  src/pcl_ros/features/moment_invariants.cpp
#  src/pcl_ros/features/normal_3d.cpp
#  src/pcl_ros/features/normal_3d_omp.cpp
#  src/pcl_ros/features/pfh.cpp
#  src/pcl_ros/features/principal_curvatures.cpp
#  src/pcl_ros/features/vfh.cpp
#)
#target_link_libraries (pcl_ros_features ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#add_dependencies(pcl_ros_features ${PROJECT_NAME}_gencfg)
#class_loader_hide_library_symbols(pcl_ros_features)
#
#
### Declare the pcl_ros_filters library
add_library(pcl_ros_filters SHARED
  src/pcl_ros/filters/extract_indices.cpp
  src/pcl_ros/filters/filter.cpp
  src/pcl_ros/filters/passthrough.cpp
  src/pcl_ros/filters/project_inliers.cpp
  src/pcl_ros/filters/radius_outlier_removal.cpp
  src/pcl_ros/filters/statistical_outlier_removal.cpp
  src/pcl_ros/filters/voxel_grid.cpp
  src/pcl_ros/filters/crop_box.cpp
)
target_link_libraries(pcl_ros_filters pcl_ros_tf ${PCL_LIBRARIES})
ament_target_dependencies(pcl_ros_filters ${dependencies})
rclcpp_components_register_node(pcl_ros_filters
  PLUGIN "pcl_ros::ExtractIndices"
  EXECUTABLE filter_extract_indices_node
)
rclcpp_components_register_node(pcl_ros_filters
  PLUGIN "pcl_ros::PassThrough"
  EXECUTABLE filter_passthrough_node
)
rclcpp_components_register_node(pcl_ros_filters
  PLUGIN "pcl_ros::ProjectInliers"
  EXECUTABLE filter_project_inliers_node
)
rclcpp_components_register_node(pcl_ros_filters
  PLUGIN "pcl_ros::RadiusOutlierRemoval"
  EXECUTABLE filter_radius_outlier_removal_node
)
rclcpp_components_register_node(pcl_ros_filters
  PLUGIN "pcl_ros::StatisticalOutlierRemoval"
  EXECUTABLE filter_statistical_outlier_removal_node
)
rclcpp_components_register_node(pcl_ros_filters
  PLUGIN "pcl_ros::CropBox"
  EXECUTABLE filter_crop_box_node
)
rclcpp_components_register_node(pcl_ros_filters
  PLUGIN "pcl_ros::VoxelGrid"
  EXECUTABLE filter_voxel_grid_node
)
class_loader_hide_library_symbols(pcl_ros_filters)
#
### Declare the pcl_ros_segmentation library
#add_library (pcl_ros_segmentation
#  src/pcl_ros/segmentation/extract_clusters.cpp
#  src/pcl_ros/segmentation/extract_polygonal_prism_data.cpp
#  src/pcl_ros/segmentation/sac_segmentation.cpp
#  src/pcl_ros/segmentation/segment_differences.cpp
#  src/pcl_ros/segmentation/segmentation.cpp
#)
#target_link_libraries(pcl_ros_segmentation pcl_ros_tf ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#add_dependencies(pcl_ros_segmentation ${PROJECT_NAME}_gencfg)
#class_loader_hide_library_symbols(pcl_ros_segmentation)
#
### Declare the pcl_ros_surface library
#add_library (pcl_ros_surface
#  src/pcl_ros/surface/surface.cpp
#  # Compilation is much faster if we include all the following CPP files in surface.cpp
#  src/pcl_ros/surface/convex_hull.cpp
#  src/pcl_ros/surface/moving_least_squares.cpp
#)
#target_link_libraries(pcl_ros_surface ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#add_dependencies(pcl_ros_surface ${PROJECT_NAME}_gencfg)
#class_loader_hide_library_symbols(pcl_ros_surface)
#
### Tools
#
add_library(pcd_to_pointcloud_lib SHARED tools/pcd_to_pointcloud.cpp)
target_link_libraries(pcd_to_pointcloud_lib
  ${PCL_LIBRARIES})
target_include_directories(pcd_to_pointcloud_lib PUBLIC
  ${PCL_INCLUDE_DIRS})
ament_target_dependencies(pcd_to_pointcloud_lib
  rclcpp
  rclcpp_components
  sensor_msgs
  pcl_conversions)
rclcpp_components_register_node(pcd_to_pointcloud_lib
  PLUGIN "pcl_ros::PCDPublisher"
  EXECUTABLE pcd_to_pointcloud)

add_library(pointcloud_to_pcd_lib SHARED tools/pointcloud_to_pcd.cpp)
target_link_libraries(pointcloud_to_pcd_lib
  ${PCL_LIBRARIES})
target_include_directories(pointcloud_to_pcd_lib PUBLIC
  ${PCL_INCLUDE_DIRS})
ament_target_dependencies(pointcloud_to_pcd_lib
  rclcpp
  rclcpp_components
  sensor_msgs
  pcl_conversions
  tf2_eigen
  tf2_ros)
rclcpp_components_register_node(pointcloud_to_pcd_lib
  PLUGIN "pcl_ros::PointCloudToPCD"
  EXECUTABLE pointcloud_to_pcd)

#
#add_executable(pointcloud_to_pcd tools/pointcloud_to_pcd.cpp)
#target_link_libraries(pointcloud_to_pcd ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#
#add_executable(bag_to_pcd tools/bag_to_pcd.cpp)
#target_link_libraries(bag_to_pcd pcl_ros_tf ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#
#add_executable(convert_pcd_to_image tools/convert_pcd_to_image.cpp)
#target_link_libraries(convert_pcd_to_image ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#
#add_executable(convert_pointcloud_to_image tools/convert_pointcloud_to_image.cpp)
#target_link_libraries(convert_pointcloud_to_image ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#
### Downloads
#
#catkin_download(table_scene_lms400.pcd http://download.ros.org/data/pcl/table_scene_lms400.pcd
#  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/samples/data
#  MD5 546b5b4822fb1de21b0cf83d41ad6683
#)
#add_custom_target(download ALL DEPENDS download_extra_data)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  add_subdirectory(tests/filters)
  #add_rostest_gtest(test_tf_message_filter_pcl tests/test_tf_message_filter_pcl.launch src/test/test_tf_message_filter_pcl.cpp)
  #target_link_libraries(test_tf_message_filter_pcl ${catkin_LIBRARIES} ${GTEST_LIBRARIES})
  #add_rostest(samples/pcl_ros/features/sample_normal_3d.launch ARGS gui:=false)
  #add_rostest(samples/pcl_ros/filters/sample_statistical_outlier_removal.launch ARGS gui:=false)
  #add_rostest(samples/pcl_ros/filters/sample_voxel_grid.launch ARGS gui:=false)
  #add_rostest(samples/pcl_ros/segmentation/sample_extract_clusters.launch ARGS gui:=false)
  #add_rostest(samples/pcl_ros/surface/sample_convex_hull.launch ARGS gui:=false)
endif()


install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS
    pcl_ros_tf
    pcd_to_pointcloud_lib
#    pcl_ros_io
#    pcl_ros_features
    pcl_ros_filters
#    pcl_ros_surface
#    pcl_ros_segmentation
    pointcloud_to_pcd_lib
#    bag_to_pcd
#    convert_pcd_to_image
#    convert_pointcloud_to_image
  EXPORT export_pcl_ros
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(DIRECTORY plugins samples
  DESTINATION share/${PROJECT_NAME})

# Export old-style CMake variables
ament_export_include_directories("include/${PROJECT_NAME}")
ament_export_libraries(pcl_ros_tf)

# Export modern CMake targets
ament_export_targets(export_pcl_ros HAS_LIBRARY_TARGET)

ament_export_dependencies(${dependencies})

ament_package()
