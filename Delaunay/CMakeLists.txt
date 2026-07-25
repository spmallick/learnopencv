cmake_minimum_required(VERSION 3.16)

project(delaunay_voronoi LANGUAGES CXX)

# OpenCV 5 requires C++17, and OpenCV 4.14 supports the same language level.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The geometry component exists only in OpenCV 5, so discover it opportunistically.
find_package(
  OpenCV CONFIG REQUIRED
  COMPONENTS core highgui imgcodecs imgproc
  OPTIONAL_COMPONENTS geometry
)

# Keep this tutorial's compatibility contract explicit.
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(
    FATAL_ERROR
    "Delaunay supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}"
  )
endif()

# Subdiv2D moved into opencv_geometry in OpenCV 5.
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5 AND NOT OpenCV_geometry_FOUND)
  message(FATAL_ERROR "OpenCV 5 requires the geometry module for Subdiv2D")
endif()

add_executable(delaunay delaunay.cpp)

# Treat third-party OpenCV headers as system headers before enabling -Werror.
target_include_directories(delaunay SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

# The OpenCV package supplies only the components resolved above.
target_link_libraries(delaunay PRIVATE ${OpenCV_LIBS})

# Bundled defaults remain valid when the executable runs from another directory.
target_compile_definitions(
  delaunay
  PRIVATE DELAUNAY_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# Strict warnings catch accidental regressions in the tutorial source itself.
if(MSVC)
  target_compile_options(delaunay PRIVATE /W4 /WX)
else()
  target_compile_options(delaunay PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)

if(BUILD_TESTING)
  # The regression executes the real CLI without opening GUI windows.
  add_test(
    NAME delaunay_regression
    COMMAND
      "${CMAKE_COMMAND}"
      "-DDELAUNAY_EXE=$<TARGET_FILE:delaunay>"
      "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/regression-test"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cpp_regression.cmake"
  )

  # Guard two important failure paths through the compiled executable.
  add_test(
    NAME delaunay_failure_safety
    COMMAND
      "${CMAKE_COMMAND}"
      "-DDELAUNAY_EXE=$<TARGET_FILE:delaunay>"
      "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/failure-tests"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cpp_failures.cmake"
  )
endif()
