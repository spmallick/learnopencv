#
# Copyright (c) 2019 Intel Corporation.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

uint32 id
string type
string rolename
CarlaEgoVehicleInfoWheel[] wheels
float32 max_rpm
float32 moi
float32 damping_rate_full_throttle
float32 damping_rate_zero_throttle_clutch_engaged
float32 damping_rate_zero_throttle_clutch_disengaged
bool use_gear_autobox
float32 gear_switch_time
float32 clutch_strength
float32 mass
float32 drag_coefficient
geometry_msgs/Vector3 center_of_mass
