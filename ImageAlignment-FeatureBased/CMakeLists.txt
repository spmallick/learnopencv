cmake_minimum_required(VERSION 3.16)

project(feature_based_image_alignment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The first lookup establishes the OpenCV major version using modules whose
# names are unchanged. OpenCV 5 renamed features2d to features and moved
# findHomography from calib3d to geometry, so request those components only
# after the version is known.
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}")
endif()

if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
  find_package(OpenCV REQUIRED COMPONENTS core features geometry imgcodecs imgproc)
else()
  find_package(OpenCV REQUIRED COMPONENTS calib3d core features2d imgcodecs imgproc)
endif()

add_executable(image_alignment align.cpp)
target_include_directories(
  image_alignment SYSTEM PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(image_alignment PRIVATE ${OpenCV_LIBS})
target_compile_definitions(
  image_alignment PRIVATE
  IMAGE_ALIGNMENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
if(MSVC)
  target_compile_options(image_alignment PRIVATE /W4 /WX)
else()
  target_compile_options(image_alignment PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME image_alignment_regression
    COMMAND
      ${CMAKE_COMMAND}
      -DEXECUTABLE=$<TARGET_FILE:image_alignment>
      -DPROJECT_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/test-output
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cpp_cli.cmake
  )
  set_tests_properties(
    image_alignment_regression PROPERTIES
    PASS_REGULAR_EXPRESSION "Alignment validation passed"
  )
endif()
