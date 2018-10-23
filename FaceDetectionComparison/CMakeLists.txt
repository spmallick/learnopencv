cmake_minimum_required(VERSION 2.8.12)

PROJECT(face_detection)

find_package( OpenCV REQUIRED )

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
