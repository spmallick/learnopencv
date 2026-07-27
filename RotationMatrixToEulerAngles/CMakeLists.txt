cmake_minimum_required(VERSION 3.16)
project(RotationMatrixToEulerAngles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core)
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}")
endif()

add_executable(rotm2euler rotm2euler.cpp)
target_include_directories(rotm2euler SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rotm2euler PRIVATE ${OpenCV_LIBS})

if(MSVC)
  target_compile_options(rotm2euler PRIVATE /W4 /WX)
else()
  target_compile_options(rotm2euler PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME rotation_matrix_validation COMMAND rotm2euler --validate)
  set_tests_properties(rotation_matrix_validation PROPERTIES
    PASS_REGULAR_EXPRESSION "VALIDATION PASSED")
endif()
