# Build configuration for the C++ tracking example.
#
# Supports OpenCV 4.x (4.9+) and OpenCV 5.x. Select a specific installation
# with -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv5 (or .../opencv4).

# 3.16 is the oldest CMake providing everything used below.
cmake_minimum_required(VERSION 3.16)
project(object_tracking LANGUAGES CXX)

# OpenCV 5 requires C++17; we require it uniformly for both majors.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Request every component used by the tracker implementations. DNN is required
# because the deep trackers load and execute ONNX networks even though this
# source does not call cv::dnn APIs directly. The contrib "tracking" module
# (KCF, CSRT) remains optional: the source detects its absence through
# opencv_modules.hpp and compiles without those two trackers.
find_package(OpenCV REQUIRED
             COMPONENTS core dnn imgproc videoio highgui video
             OPTIONAL_COMPONENTS tracking)

# Reject unsupported versions early with a clear message. 4.9 is the floor
# because TrackerVit first shipped in OpenCV 4.9.0.
if(OpenCV_VERSION VERSION_LESS 4.9 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.9+ and 5.x, not ${OpenCV_VERSION}")
endif()

add_executable(object_tracking object_tracking.cpp)

# Target-scoped usage requirements only; no global include/link state.
target_include_directories(object_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(object_tracking PRIVATE ${OpenCV_LIBS})

# Bake the shared models directory (next to the project sources) into the
# binary so it works from any current directory; --models-dir overrides it.
target_compile_definitions(object_tracking PRIVATE
  MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models")

# Strict warnings; MSVC gets its closest equivalents.
if(MSVC)
  target_compile_options(object_tracking PRIVATE /W4 /WX)
else()
  target_compile_options(object_tracking PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
endif()

# ---------------------------------------------------------------------------
# CTest regressions use CMake wrappers rather than relying only on a process
# exit code. Each wrapper checks the success marker, the exact output manifest,
# nonempty artifacts, JSON fields, and the readability of both generated AVIs.
# ---------------------------------------------------------------------------
enable_testing()

# Helper: register one validation test for a tracker name.
function(add_tracker_test name)
  add_test(NAME validate_${name}
    COMMAND ${CMAKE_COMMAND}
      -DEXECUTABLE=$<TARGET_FILE:object_tracking>
      -DTRACKER=${name}
      -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_out_${name}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunTrackerValidation.cmake)
  set_tests_properties(validate_${name} PROPERTIES
    # DNN inference can become pathologically slow when multiple trackers
    # compete for the same CPU. Keep validation deterministic even under
    # ctest -j or CTEST_PARALLEL_LEVEL, while retaining a bounded timeout.
    RUN_SERIAL TRUE
    TIMEOUT 300)
endfunction()

# MIL ships with every supported OpenCV build: always test it.
add_tracker_test(mil)

# KCF and CSRT exist only when the contrib tracking module was found.
if(TARGET opencv_tracking OR ";${OpenCV_LIBS};" MATCHES ";opencv_tracking;")
  add_tracker_test(kcf)
  add_tracker_test(csrt)
endif()

# DNN trackers are tested when their model files are present at configure
# time (fetch them with: python3 ../download_models.py).
set(_models_dir "${CMAKE_CURRENT_SOURCE_DIR}/../models")
if(EXISTS "${_models_dir}/nanotrack_backbone_sim.onnx" AND
   EXISTS "${_models_dir}/nanotrack_head_sim.onnx")
  add_tracker_test(nanotrack)
endif()
if(EXISTS "${_models_dir}/object_tracking_vittrack_2023sep.onnx")
  add_tracker_test(vittrack)
endif()
if(EXISTS "${_models_dir}/dasiamrpn_model.onnx" AND
   EXISTS "${_models_dir}/dasiamrpn_kernel_cls1.onnx" AND
   EXISTS "${_models_dir}/dasiamrpn_kernel_r1.onnx")
  add_tracker_test(dasiamrpn)
endif()

# Helper: register a command expected to fail. The wrapper requires both a
# normal numeric nonzero exit and the case-specific diagnostic text.
function(add_failure_test name arguments expected_pattern)
  add_test(NAME ${name}
    COMMAND ${CMAKE_COMMAND}
      -DEXECUTABLE=$<TARGET_FILE:object_tracking>
      "-DTEST_ARGUMENTS=${arguments}"
      "-DEXPECTED_PATTERN=${expected_pattern}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunExpectedFailure.cmake)
endfunction()

# Missing media, unknown options, and invalid typed values exercise distinct
# clean-error paths before any validation-generated input is needed.
add_failure_test(
  missing_input_fails
  "--tracker=mil|--input=no_such_file.mp4|--bbox=10,10,40,40|--no-display"
  "error: Cannot open input")
add_failure_test(
  unknown_option_fails
  "--not-an-option=1"
  "Unknown option")
add_failure_test(
  invalid_max_frames_fails
  "--max-frames=not-an-integer"
  "can not convert")
add_failure_test(
  negative_max_frames_fails
  "--max-frames=-1"
  "--max-frames must be zero or positive")
add_failure_test(
  unknown_tracker_fails
  "--tracker=goturn"
  "unknown tracker name")
add_failure_test(
  output_path_fails
  "--tracker=mil|--validate|--no-display|--output-dir=${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt"
  "error:")
add_failure_test(
  truncated_validation_fails
  "--tracker=mil|--validate|--no-display|--max-frames=2|--output-dir=${CMAKE_CURRENT_BINARY_DIR}/test_out_truncated"
  "VALIDATION FAILED")

# Bounding-box syntax and geometry checks need a real frame. Depend on the MIL
# validation test, which deterministically creates this 640x360 input clip.
set(_mil_test_clip
    "${CMAKE_CURRENT_BINARY_DIR}/test_out_mil/synthetic_clip.avi")
add_failure_test(
  bbox_trailing_junk_fails
  "--tracker=mil|--input=${_mil_test_clip}|--bbox=20,148,64,64junk|--no-display"
  "expected exactly x,y,w,h integers")
add_failure_test(
  bbox_nonpositive_size_fails
  "--tracker=mil|--input=${_mil_test_clip}|--bbox=20,148,0,64|--no-display"
  "width and height must be positive")
add_failure_test(
  bbox_outside_frame_fails
  "--tracker=mil|--input=${_mil_test_clip}|--bbox=620,340,40,40|--no-display"
  "rectangle must lie inside the first frame")
add_failure_test(
  headless_without_bbox_fails
  "--tracker=mil|--input=${_mil_test_clip}|--no-display"
  "--bbox is required when --no-display is set")
set_tests_properties(
  bbox_trailing_junk_fails bbox_nonpositive_size_fails bbox_outside_frame_fails
  headless_without_bbox_fails
  PROPERTIES DEPENDS validate_mil)
