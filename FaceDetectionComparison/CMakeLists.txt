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

find_package(dlib QUIET)
if(dlib_FOUND)
  message(STATUS "Found system linstance of DLib")
else()
  message(STATUS "Use own instance of DLib")
  set(dlib_FOUND 1)
  include(./dlib/dlib/cmake)
endif()

MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
  IF(dlib_FOUND)
    TARGET_LINK_LIBRARIES(${name} dlib::dlib)
  ENDIF()
ENDMACRO()

add_example(face_detection_opencv_haar)
if(${OpenCV_VERSION} VERSION_GREATER 3.4)
  add_example(face_detection_opencv_dnn)
endif()

if(dlib_FOUND AND ${OpenCV_VERSION} VERSION_LESS 3.0)
  add_example(face_detection_dlib_hog)
  add_example(face_detection_dlib_mmod)
endif()

add_example(run-all)
