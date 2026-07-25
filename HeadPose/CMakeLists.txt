cmake_minimum_required(VERSION 3.16)
project(head_pose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui calib)
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui calib3d)
endif()

add_executable(head_pose headPose.cpp)
target_link_libraries(head_pose PRIVATE ${OpenCV_LIBS})
