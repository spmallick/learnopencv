#
# Copyright (c) 2019 Intel Corporation.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# A lane invasion event

std_msgs/Header header

int32[] crossed_lane_markings

int32 LANE_MARKING_OTHER=0
int32 LANE_MARKING_BROKEN=1
int32 LANE_MARKING_SOLID=2

