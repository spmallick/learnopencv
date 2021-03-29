cmake_minimum_required(VERSION 3.15)
project(OpticalFlow)
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 14)

add_executable(OpticalFlow main.cpp)
target_link_libraries( OpticalFlow ${OpenCV_LIBS} )
