cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SuperResolution)

if(MSVC)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SuperResolution super_res.cpp)

target_link_libraries(SuperResolution ${OpenCV_LIBS})

set_property(TARGET SuperResolution PROPERTY CXX_STANDARD 14)
