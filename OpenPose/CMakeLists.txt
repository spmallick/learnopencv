cmake_minimum_required(VERSION 3.16)

project(OpenPoseMediaPipe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
  OpenCV REQUIRED
  COMPONENTS core dnn highgui imgcodecs imgproc videoio
)
if(OpenCV_VERSION VERSION_LESS 4.10 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "These examples require OpenCV 4.10+ or 5.x, not ${OpenCV_VERSION}")
endif()

function(add_pose_example target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(
    ${target} PRIVATE OPENPOSE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(
      ${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

add_pose_example(OpenPoseImage OpenPoseImage.cpp)
add_pose_example(OpenPoseVideo OpenPoseVideo.cpp)

include(CTest)
if(BUILD_TESTING)
  set(POSE_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/models/pose_estimation_mediapipe_2023mar.onnx")
  if(NOT EXISTS "${POSE_MODEL}")
    message(FATAL_ERROR
      "Model missing at ${POSE_MODEL}; run download_models.py before CTest.")
  endif()

  add_test(
    NAME openpose_image_regression
    COMMAND OpenPoseImage
      --input "${CMAKE_CURRENT_SOURCE_DIR}/single.jpeg"
      --model "${POSE_MODEL}"
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output/image"
      --no-display
      --validate
  )
  set_tests_properties(
    openpose_image_regression
    PROPERTIES PASS_REGULAR_EXPRESSION
      "VALIDATION PASSED: landmarks=33 visible=33 edges=35")

  add_test(
    NAME openpose_video_regression
    COMMAND OpenPoseVideo
      --input "${CMAKE_CURRENT_SOURCE_DIR}/sample_video.mp4"
      --model "${POSE_MODEL}"
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output/video"
      --max-frames 2
      --no-display
      --validate
  )
  set_tests_properties(
    openpose_video_regression
    PROPERTIES PASS_REGULAR_EXPRESSION "VALIDATION PASSED: frames=2")
endif()
