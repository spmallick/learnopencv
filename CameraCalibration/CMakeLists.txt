cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 14)

PROJECT(cameracalib)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
ENDMACRO()

add_example(cameraCalibration)
