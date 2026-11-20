cmake_minimum_required(VERSION 3.18)
project(robust_image_alignment LANGUAGES CXX)

# Modern filesystem/optional types keep the example independent of Python.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_ALIGNMENT_APP "Build native feature matching and notebook workflows" ON)
option(BUILD_TESTING "Build meaningful geometry and command-line regressions" ON)

# This companion deliberately targets the released version used by the article.
find_package(OpenCV 5.1.0 EXACT REQUIRED COMPONENTS core imgproc imgcodecs features geometry video)

function(alignment_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

add_library(alignment_geometry cpp/alignment.cpp)
target_include_directories(alignment_geometry PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
target_include_directories(alignment_geometry SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(alignment_geometry PUBLIC ${OpenCV_LIBS})
alignment_warnings(alignment_geometry)

if(BUILD_ALIGNMENT_APP)
  include(cpp/verify_models.cmake)
  include(cpp/onnxruntime.cmake)
  add_library(alignment_features cpp/feature_matcher.cpp)
  target_link_libraries(alignment_features PUBLIC alignment_geometry ONNXRuntime::ONNXRuntime)
  alignment_warnings(alignment_features)
  add_executable(robust_image_alignment_cpp cpp/main.cpp cpp/workflows.cpp)
  target_link_libraries(robust_image_alignment_cpp PRIVATE alignment_features)
  target_compile_definitions(robust_image_alignment_cpp PRIVATE
    ROBUST_ALIGNMENT_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  set_target_properties(robust_image_alignment_cpp PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")
  alignment_warnings(robust_image_alignment_cpp)
endif()

if(BUILD_TESTING)
  enable_testing()
  add_executable(alignment_geometry_tests cpp/tests/geometry_tests.cpp)
  target_link_libraries(alignment_geometry_tests PRIVATE alignment_geometry)
  alignment_warnings(alignment_geometry_tests)
  add_test(NAME geometry_regressions COMMAND alignment_geometry_tests)
  set_tests_properties(geometry_regressions PROPERTIES PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
  if(TARGET robust_image_alignment_cpp)
    add_executable(alignment_sift_resize_tests cpp/tests/sift_resize_tests.cpp)
    target_link_libraries(alignment_sift_resize_tests PRIVATE alignment_features)
    set_target_properties(alignment_sift_resize_tests PROPERTIES
      BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")
    alignment_warnings(alignment_sift_resize_tests)
    add_test(NAME sift_resize_pixel_centers COMMAND alignment_sift_resize_tests)
    set_tests_properties(sift_resize_pixel_centers PROPERTIES PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
    # Each run starts from the build tree, exercising source-independent input resolution.
    foreach(method sift aliked-nn lightglue)
      foreach(estimator ransac magsac)
        set(case_name "cli_${method}_${estimator}")
        add_test(NAME "${case_name}" COMMAND robust_image_alignment_cpp
          --reference "${CMAKE_CURRENT_SOURCE_DIR}/assets/downloads/opencv/box.png"
          --moving "${CMAKE_CURRENT_SOURCE_DIR}/assets/downloads/opencv/box_in_scene.png"
          --method "${method}" --estimator "${estimator}" --device cpu
          --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output/${case_name}"
          --no-display --validate)
        set_tests_properties("${case_name}" PROPERTIES PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
      endforeach()
    endforeach()
    add_test(NAME native_notebook_demos COMMAND robust_image_alignment_cpp
      --demo-all --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output/demos" --no-display --validate)
    set_tests_properties(native_notebook_demos PROPERTIES PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
    add_test(NAME native_synthetic_study COMMAND robust_image_alignment_cpp
      --synthetic-study --study-seeds 3
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output/synthetic" --no-display --validate)
    set_tests_properties(native_synthetic_study PROPERTIES PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
    add_test(NAME cli_errors_and_reuse COMMAND "${CMAKE_COMMAND}"
      "-DEXECUTABLE=$<TARGET_FILE:robust_image_alignment_cpp>"
      "-DPROJECT_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-output/cli-regressions"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/check_cli.cmake")
    set_tests_properties(cli_errors_and_reuse PROPERTIES PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
  endif()
endif()
