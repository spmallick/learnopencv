find_package(ament_cmake_pytest REQUIRED)

# build dummy publisher node and component
add_library(dummy_topics SHARED
  dummy_topics.cpp
)
target_link_libraries(dummy_topics ${PCL_LIBRARIES})
ament_target_dependencies(dummy_topics
  rclcpp
  rclcpp_components
  pcl_conversions
  sensor_msgs
  PCL
)

# generate test ament index to locate dummy_topics comonent from tests
set(components "")
set(components "${components}pcl_ros_tests_filters::DummyTopics;$<TARGET_FILE:dummy_topics>\n")
file(GENERATE
OUTPUT
"${CMAKE_CURRENT_BINARY_DIR}/test_ament_index/share/ament_index/resource_index/rclcpp_components/pcl_ros_tests_filters"
CONTENT "${components}")

# test components
ament_add_pytest_test(test_pcl_ros::ExtractIndices
  test_filter_component.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_PLUGIN=pcl_ros::ExtractIndices
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_pcl_ros::PassThrough
  test_filter_component.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_PLUGIN=pcl_ros::PassThrough
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_pcl_ros::ProjectInliers
  test_filter_component.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_PLUGIN=pcl_ros::ProjectInliers
      PARAMETERS={'model_type':0}
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_pcl_ros::RadiusOutlierRemoval
  test_filter_component.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_PLUGIN=pcl_ros::RadiusOutlierRemoval
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_pcl_ros::StatisticalOutlierRemoval
  test_filter_component.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_PLUGIN=pcl_ros::StatisticalOutlierRemoval
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_pcl_ros::CropBox
  test_filter_component.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_PLUGIN=pcl_ros::CropBox
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_pcl_ros::VoxelGrid
  test_filter_component.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_PLUGIN=pcl_ros::VoxelGrid
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)

# test executables
ament_add_pytest_test(test_filter_extract_indices_node
  test_filter_executable.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_EXECUTABLE=filter_extract_indices_node
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_filter_passthrough_node
  test_filter_executable.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_EXECUTABLE=filter_passthrough_node
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_filter_project_inliers_node
  test_filter_executable.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_EXECUTABLE=filter_project_inliers_node
      PARAMETERS={'model_type':0}
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_filter_radius_outlier_removal_node
  test_filter_executable.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_EXECUTABLE=filter_radius_outlier_removal_node
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_filter_statistical_outlier_removal_node
  test_filter_executable.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_EXECUTABLE=filter_statistical_outlier_removal_node
    APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_filter_crop_box_node
  test_filter_executable.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_EXECUTABLE=filter_crop_box_node
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
ament_add_pytest_test(test_filter_voxel_grid_node
  test_filter_executable.py
  ENV DUMMY_PLUGIN=pcl_ros_tests_filters::DummyTopics
      FILTER_EXECUTABLE=filter_voxel_grid_node
  APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/test_ament_index
)
