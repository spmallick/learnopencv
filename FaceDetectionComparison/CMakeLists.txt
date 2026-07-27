cmake_minimum_required(VERSION 3.16)

project(FaceDetectionComparison LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
  OpenCV REQUIRED
  COMPONENTS core dnn highgui imgcodecs imgproc objdetect videoio
)
if(OpenCV_VERSION VERSION_LESS 4.10 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "These examples require OpenCV 4.10+ or 5.x, not ${OpenCV_VERSION}")
endif()

# dlib is an optional historical baseline. It is never fetched or unpacked by
# CMake, and the primary YuNet targets remain independent of it.
find_package(dlib QUIET CONFIG)

function(configure_face_target target)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(
    ${target} PRIVATE FACE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(
      ${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

add_executable(
  face_detection_opencv_dnn
  face_detection_opencv_dnn.cpp
)
configure_face_target(face_detection_opencv_dnn)

add_executable(run-all run-all.cpp)
configure_face_target(run-all)

if(dlib_FOUND)
  target_link_libraries(run-all PRIVATE dlib::dlib)
  target_compile_definitions(run-all PRIVATE FACE_WITH_DLIB=1)

  add_executable(face_detection_dlib_hog face_detection_dlib_hog.cpp)
  configure_face_target(face_detection_dlib_hog)
  target_link_libraries(face_detection_dlib_hog PRIVATE dlib::dlib)
  target_compile_definitions(face_detection_dlib_hog PRIVATE FACE_WITH_DLIB=1)
  message(STATUS "Optional dlib HOG targets enabled")
else()
  message(STATUS "dlib not found; optional HOG targets are disabled")
endif()

# CascadeClassifier moved out of the OpenCV 5 core objdetect API. Preserve the
# historical executable only where the header and implementation still exist.
if(OpenCV_VERSION VERSION_LESS 5)
  add_executable(
    face_detection_opencv_haar
    face_detection_opencv_haar.cpp
  )
  configure_face_target(face_detection_opencv_haar)
else()
  message(STATUS "OpenCV 5 detected; optional Haar C++ target is disabled")
endif()

include(CTest)
if(BUILD_TESTING)
  set(YUNET_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/models/face_detection_yunet_2026may.onnx")
  if(NOT EXISTS "${YUNET_MODEL}")
    message(FATAL_ERROR
      "Model missing at ${YUNET_MODEL}; run download_models.py before CTest.")
  endif()

  add_test(
    NAME face_yunet_video_regression
    COMMAND face_detection_opencv_dnn
      --input "${CMAKE_CURRENT_SOURCE_DIR}/videos/baby.mp4"
      --mode video
      --model "${YUNET_MODEL}"
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output/yunet"
      --max-frames 2
      --no-display
      --validate
  )
  set_tests_properties(
    face_yunet_video_regression
    PROPERTIES PASS_REGULAR_EXPRESSION
      "FACE VIDEO RESULT: detector=YuNet frames=2 total_faces=4")

  add_test(
    NAME face_comparison_video_regression
    COMMAND run-all
      --input "${CMAKE_CURRENT_SOURCE_DIR}/videos/baby.mp4"
      --mode video
      --model "${YUNET_MODEL}"
      --detectors yunet
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output/comparison"
      --max-frames 2
      --no-display
      --validate
  )
  set_tests_properties(
    face_comparison_video_regression
    PROPERTIES PASS_REGULAR_EXPRESSION
      "COMPARISON VIDEO RESULT: frames=2 YuNet=4")
endif()
