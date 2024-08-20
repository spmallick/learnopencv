import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from os.path import expanduser


def generate_launch_description():

  # Configure environment
  stdout_linebuf_envvar = SetEnvironmentVariable('RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED', '1')
  stdout_colorized_envvar = SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1')

  # Simulated time
  use_sim_time = LaunchConfiguration('use_sim_time', default='true')

  # Nodes Configurations
  config_file = os.path.join(get_package_share_directory('lego_loam_sr'), 'config', 'loam_config.yaml')
  rviz_config = "LeGO-LOAM/rviz/pink_scan.rviz" #"/home/somusan/OpencvUni/opencvblog/robotics-series/intro-to-lidar-slam/src/LeGO-LOAM-SR/LeGO-LOAM/rviz/test.rviz"  #"LeGO-LOAM/rviz/test.rviz"
  # rviz_config = os.path.join(get_package_share_directory('lego_loam_sr'), 'rviz', 'test.rviz')

  # Tf transformations
  transform_map = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='camera_init_to_map',
    arguments=['0', '0', '0', '1.570795', '0', '1.570795', 'map', 'camera_init'],
  )

  transform_camera = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='base_link_to_camera',
    arguments=['0', '0', '0', '-1.570795', '-1.570795', '0', 'camera', 'base_link'],
  )

  transform_velodyne = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='velodyne_to_base_link',
    arguments=['0', '0', '0', '0', '0', '0','base_link','velodyne'],
  )

  # LeGO-LOAM
  lego_loam_node = Node(
    package='lego_loam_sr',
    executable='lego_loam_sr',
    output='screen',
    parameters=[config_file],
    # messages published to the topic /velodyne_points will be remapped (or redirected) to the topic /lidar_points
    remappings=[('/lidar_points', '/velodyne_points')],
  )

  # Rviz
  rviz_node = Node(
    package='rviz2',
    executable='rviz2',
    name='rviz2',
    arguments=['-d', rviz_config],
    output='screen'
  )

  ld = LaunchDescription()
  # Set environment variables
  ld.add_action(stdout_linebuf_envvar)
  ld.add_action(stdout_colorized_envvar)
  # Add nodes
  ld.add_action(lego_loam_node)
  ld.add_action(transform_map)
  ld.add_action(transform_camera)
  ld.add_action(transform_velodyne)
  ld.add_action(rviz_node)

  return ld
