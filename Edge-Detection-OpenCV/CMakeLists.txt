cmake_minimum_required(VERSION 3.16)
project(edge_detection_opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)

add_executable(edge_detection edge_detection.cpp)
target_link_libraries(edge_detection PRIVATE ${OpenCV_LIBS})
target_include_directories(edge_detection PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MSVC)
  target_compile_options(edge_detection PRIVATE /W4 /WX)
else()
  target_compile_options(edge_detection PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME edge_detection_self_test COMMAND edge_detection --self-test)
endif()
