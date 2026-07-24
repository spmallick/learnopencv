cmake_minimum_required(VERSION 3.16)
project(camera_calibration LANGUAGES CXX)

# Both OpenCV 4.14 and OpenCV 5 require no compiler extensions for this
# tutorial, so a strict C++17 target keeps the example portable.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Discover the package version through core before choosing the generation-
# specific calibration components. Unlike an unqualified package lookup, this
# first call does not populate OpenCV_LIBS with every installed module.
find_package(OpenCV CONFIG REQUIRED COMPONENTS core)

# Reject versions outside the tutorial's tested compatibility contract.
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR "This example supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}")
endif()

# Require only the shared modules plus the calibration modules used by the
# selected major version. OpenCV_LIBS retains OpenCV's own component-to-target
# mapping, including opencv_world builds.
if(OpenCV_VERSION VERSION_LESS 5)
  find_package(
    OpenCV CONFIG REQUIRED
    COMPONENTS core highgui imgcodecs imgproc calib3d
  )
else()
  find_package(
    OpenCV CONFIG REQUIRED
    COMPONENTS core highgui imgcodecs imgproc calib geometry objdetect
  )
endif()

# Treat compiler diagnostics in tutorial-owned code as build failures.
function(enable_strict_warnings target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /WX)
  else()
    target_compile_options(
      ${target_name} PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
endfunction()

# Give both tutorial executables identical dependencies and asset resolution.
function(add_calibration_example name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(
    ${name} PRIVATE TUTORIAL_IMAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/images"
  )
  enable_strict_warnings(${name})
endfunction()

# Build the calibration-only and calibration-plus-undistortion lessons.
add_calibration_example(cameraCalibration)
add_calibration_example(cameraCalibrationWithUndistortion)

# CTest exercises both real programs and a fixture-specific numerical oracle.
include(CTest)
if(BUILD_TESTING)
  add_executable(
    camera_calibration_regression
    tests/test_camera_calibration.cpp
  )
  target_compile_features(camera_calibration_regression PRIVATE cxx_std_17)
  target_include_directories(
    camera_calibration_regression
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(
    camera_calibration_regression PRIVATE ${OpenCV_LIBS}
  )
  enable_strict_warnings(camera_calibration_regression)

  add_test(
    NAME camera_calibration_regression
    COMMAND cameraCalibration --no-display --validate
  )
  set_tests_properties(
    camera_calibration_regression PROPERTIES PASS_REGULAR_EXPRESSION "Validation passed"
  )

  add_test(
    NAME camera_undistortion_regression
    COMMAND
      ${CMAKE_COMMAND}
      "-DPROGRAM=$<TARGET_FILE:cameraCalibrationWithUndistortion>"
      "-DOUTPUT_ROOT=${CMAKE_CURRENT_BINARY_DIR}/test-output"
      "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-output/camera-undistortion"
      "-DEXPECTED_MARKER=Validation passed"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_undistortion_cli.cmake"
  )

  add_test(
    NAME camera_calibration_fixture_quality
    COMMAND
      camera_calibration_regression
      "${CMAKE_CURRENT_SOURCE_DIR}/images"
  )

  # A missing input must fail through the public CLI instead of reaching
  # calibration with an empty set.
  add_test(
    NAME camera_calibration_rejects_missing_images
    COMMAND
      ${CMAKE_COMMAND}
      "-DPROGRAM=$<TARGET_FILE:cameraCalibration>"
      "-DARGUMENT_1=--images"
      "-DARGUMENT_2=${CMAKE_CURRENT_BINARY_DIR}/missing-*.jpg"
      "-DARGUMENT_3=--no-display"
      "-DEXPECTED_ERROR=Error: No calibration images matched"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_failure.cmake"
  )

  # The second executable has its own parser, so verify its unknown-option
  # failure path independently.
  add_test(
    NAME camera_undistortion_rejects_unknown_argument
    COMMAND
      ${CMAKE_COMMAND}
      "-DPROGRAM=$<TARGET_FILE:cameraCalibrationWithUndistortion>"
      "-DARGUMENT_1=--not-an-option"
      "-DEXPECTED_ERROR=Error: Unknown argument: --not-an-option"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_cli_failure.cmake"
  )
endif()
