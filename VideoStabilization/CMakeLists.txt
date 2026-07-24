cmake_minimum_required(VERSION 3.16)
project(VideoStabilization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED
  COMPONENTS calib3d core highgui imgproc video videoio)
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example targets OpenCV 4.14 and 5.0, not ${OpenCV_VERSION}")
endif()

add_executable(video_stabilization video_stabilization.cpp)
target_include_directories(video_stabilization PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(video_stabilization PRIVATE ${OpenCV_LIBS})
target_compile_definitions(video_stabilization
  PRIVATE VIDEO_STABILIZATION_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
  target_compile_options(video_stabilization PRIVATE /W4 /WX)
else()
  target_compile_options(video_stabilization
    PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME video_stabilization_regression
    COMMAND video_stabilization
      --input "${CMAKE_CURRENT_SOURCE_DIR}/video.mp4"
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output"
      --no-display
      --validate)
  set_tests_properties(video_stabilization_regression PROPERTIES
    PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
endif()
