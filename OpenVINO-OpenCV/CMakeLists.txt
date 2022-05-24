cmake_minimum_required(VERSION 2.8.12)

PROJECT(openvino_opencv)

#set(OpenCV_DIR <specify your path to the opencv installation directory>installations/OpenCV4/lib/cmake/opencv4/)
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories( ${OpenCV_INCLUDE_DIRS})

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
ENDMACRO()

add_example(image-classification)
