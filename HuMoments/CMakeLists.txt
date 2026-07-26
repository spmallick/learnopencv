cmake_minimum_required(VERSION 3.16)

project(HuMomentsExamples LANGUAGES CXX)

find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS 4.8 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR "This example supports OpenCV 4.8 through 5.x")
endif()
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
  find_package(OpenCV REQUIRED COMPONENTS core geometry imgcodecs imgproc)
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
endif()

function(add_hu_example target)
  add_executable(${target} ${target}.cpp)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

add_hu_example(HuMoments)
add_hu_example(shapeMatcher)
target_compile_definitions(
  shapeMatcher PRIVATE HU_MOMENTS_IMAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/images"
)

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME hu_moments_cpp
    COMMAND HuMoments "${CMAKE_CURRENT_SOURCE_DIR}/images/K0.png"
  )
  set_tests_properties(
    hu_moments_cpp PROPERTIES PASS_REGULAR_EXPRESSION "2\\.78871"
  )

  add_test(NAME shape_matcher_cpp COMMAND shapeMatcher)
  set_tests_properties(
    shape_matcher_cpp PROPERTIES
    PASS_REGULAR_EXPRESSION "S0\\.png and K0\\.png: 0\\.114402"
  )
endif()
