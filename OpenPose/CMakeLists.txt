cmake_minimum_required(VERSION 2.8.12)

PROJECT(openPose)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
ENDMACRO()

add_example(OpenPoseImage)
add_example(OpenPoseVideo)
