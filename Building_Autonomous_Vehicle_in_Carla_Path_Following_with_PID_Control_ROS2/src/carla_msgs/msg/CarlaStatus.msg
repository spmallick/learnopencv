#
# Copyright (c) 2019 Intel Corporation.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#

uint64 frame                  # frame number

float32 fixed_delta_seconds   # duration of one frame
bool synchronous_mode         # carla is in synchronous mode
bool synchronous_mode_running # true: running, false: paused