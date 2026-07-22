cmake_minimum_required(VERSION 3.16)
project(contour_approximations LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR "This example supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}")
endif()

add_executable(contour_approximations contour_approx.cpp)
target_include_directories(contour_approximations PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(contour_approximations PRIVATE ${OpenCV_LIBS})
target_compile_definitions(
  contour_approximations PRIVATE TUTORIAL_INPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../input"
)

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME contour_approximations_regression
    COMMAND contour_approximations --no-display --validate
            --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output"
  )
  set_tests_properties(
    contour_approximations_regression PROPERTIES PASS_REGULAR_EXPRESSION "Validation passed"
  )
endif()
