cmake_minimum_required(VERSION 3.16)
project(HoughTransformTutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}")
endif()

function(add_hough_example target source)
  add_executable(${target} ${source})
  target_include_directories(${target}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(${target}
    SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(${target} PRIVATE
    HOUGH_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

add_hough_example(hough_lines hough_lines.cpp)
add_hough_example(hough_circles hough_circles.cpp)

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME hough_lines_validation
    COMMAND hough_lines
      --validate
      --no-display
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output")
  set_tests_properties(hough_lines_validation PROPERTIES
    PASS_REGULAR_EXPRESSION "VALIDATION PASSED")

  add_test(
    NAME hough_circles_validation
    COMMAND hough_circles
      --validate
      --no-display
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output")
  set_tests_properties(hough_circles_validation PROPERTIES
    PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
endif()
