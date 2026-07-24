# Build configuration for the C++ tracking example.
#
# Supports OpenCV 4.x (4.8+) and OpenCV 5.x. Select a specific installation
# with -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv5 (or .../opencv4).

# 3.16 is the oldest CMake providing everything used below.
cmake_minimum_required(VERSION 3.16)
project(object_tracking LANGUAGES CXX)

# OpenCV 5 requires C++17; we require it uniformly for both majors.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Request only the components this example actually uses. The contrib
# "tracking" module (KCF, CSRT) is optional: the source detects its absence
# through opencv_modules.hpp and compiles without those two trackers.
find_package(OpenCV REQUIRED
             COMPONENTS core imgproc videoio highgui video
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
# CTest regressions: run the validation mode headlessly into a build-local
# output directory and require the explicit success marker.
# ---------------------------------------------------------------------------
enable_testing()

# Helper: register one validation test for a tracker name.
function(add_tracker_test name)
  add_test(NAME validate_${name}
           COMMAND object_tracking --tracker=${name} --validate --no-display
                   --output-dir=${CMAKE_CURRENT_BINARY_DIR}/test_out_${name})
  set_tests_properties(validate_${name} PROPERTIES
    PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
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

# The error-handling contract is part of the example: a missing input file
# must fail with a clean nonzero exit and a readable message.
add_test(NAME missing_input_fails
         COMMAND object_tracking --tracker=mil --input=no_such_file.mp4
                 --bbox=10,10,40,40 --no-display)
set_tests_properties(missing_input_fails PROPERTIES
  PASS_REGULAR_EXPRESSION "Cannot open input")
