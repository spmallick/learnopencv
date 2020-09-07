cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

PROJECT(face_detection)

find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS})
include(./dlib/dlib/cmake)

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS} dlib::dlib)
ENDMACRO()

add_example(face_detection_opencv_haar)
add_example(face_detection_opencv_dnn)
add_example(face_detection_dlib_hog)
add_example(face_detection_dlib_mmod)
add_example(run-all)
