cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Illuminate)

if(MSVC)
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Illuminate brighten.cpp guidedfilter.h guidedfilter.cpp)

target_link_libraries(Illuminate ${OpenCV_LIBS})

set_property(TARGET Illuminate PROPERTY CXX_STANDARD 14)
