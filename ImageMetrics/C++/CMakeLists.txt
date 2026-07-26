cmake_minimum_required(VERSION 3.16)

project(BRISQUEQualityExample LANGUAGES CXX)

find_package(
  OpenCV REQUIRED
  COMPONENTS core imgcodecs imgproc ml quality
)
if(OpenCV_VERSION VERSION_LESS 4.8 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR "This example supports OpenCV 4.8 through 5.x")
endif()

add_executable(brisquequality main.cpp)
target_compile_features(brisquequality PRIVATE cxx_std_17)
target_link_libraries(brisquequality PRIVATE ${OpenCV_LIBS})
target_compile_definitions(
  brisquequality
  PRIVATE
    BRISQUE_DEFAULT_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/../models/brisque_model_live.yml"
    BRISQUE_DEFAULT_RANGE="${CMAKE_CURRENT_SOURCE_DIR}/../models/brisque_range_live.yml"
)

if(MSVC)
  target_compile_options(brisquequality PRIVATE /W4)
else()
  target_compile_options(brisquequality PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME brisque_cpp_regression
    COMMAND
      brisquequality
      "${CMAKE_CURRENT_SOURCE_DIR}/../Images/original-scaled-image.jpg"
  )
  set_tests_properties(
    brisque_cpp_regression PROPERTIES
    PASS_REGULAR_EXPRESSION "BRISQUE score: 20\\.27"
  )
endif()
