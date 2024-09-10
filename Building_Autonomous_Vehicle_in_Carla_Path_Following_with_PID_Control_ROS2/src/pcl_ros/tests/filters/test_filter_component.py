#
# Copyright (c) 2022  Carnegie Mellon University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Carnegie Mellon University nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import ast
import os
import unittest

import launch
import launch.actions
import launch_ros.actions
import launch_testing.actions
import launch_testing.markers
from launch_testing_ros import WaitForTopics
import pytest
from sensor_msgs.msg import PointCloud2


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    dummy_plugin = os.getenv('DUMMY_PLUGIN')
    filter_plugin = os.getenv('FILTER_PLUGIN')
    parameters = ast.literal_eval(os.getenv('PARAMETERS')) if 'PARAMETERS' in os.environ else {}

    print(parameters)

    return launch.LaunchDescription([
        launch_ros.actions.ComposableNodeContainer(
            name='filter_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                launch_ros.descriptions.ComposableNode(
                    package='pcl_ros_tests_filters',
                    plugin=dummy_plugin,
                    name='dummy_publisher',
                ),
                launch_ros.descriptions.ComposableNode(
                    package='pcl_ros',
                    plugin=filter_plugin,
                    name='filter_node',
                    remappings=[('/input', '/point_cloud2')],
                    parameters=[parameters],
                ),
            ],
            output='screen',
        ),
        launch_testing.actions.ReadyToTest()
    ])


class TestFilter(unittest.TestCase):

    def test_filter_output(self):
        wait_for_topics = WaitForTopics([('output', PointCloud2)], timeout=5.0)
        assert wait_for_topics.wait()
        assert 'output' in wait_for_topics.topics_received(), "Didn't receive message"
        wait_for_topics.shutdown()
