
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 14)

PROJECT(cameracalib)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
ENDMACRO()

add_example(disparity_params_gui)
add_example(disparity2depth_calib)
add_example(obstacle_avoidance)
