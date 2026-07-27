cmake_minimum_required(VERSION 3.16)
project(ColorSpacesTutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}")
endif()

function(add_color_spaces_example target source)
  add_executable(${target} ${source})
  target_include_directories(${target}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(${target}
    SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(${target} PRIVATE
    COLOR_SPACES_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

add_color_spaces_example(interactiveColorDetect interactiveColorDetect.cpp)
add_color_spaces_example(interactiveColorSegment interactiveColorSegment.cpp)

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME color_detect_validation
    COMMAND interactiveColorDetect
      --validate
      --no-display
      --output "${CMAKE_CURRENT_BINARY_DIR}/test-output/color-values.png")
  set_tests_properties(color_detect_validation PROPERTIES
    PASS_REGULAR_EXPRESSION "VALIDATION PASSED")

  add_test(
    NAME color_segment_validation
    COMMAND interactiveColorSegment
      --validate
      --no-display
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output")
  set_tests_properties(color_segment_validation PROPERTIES
    PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
endif()
