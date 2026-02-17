cmake_minimum_required(VERSION 3.5.1)

project(sample)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB srcs *.cpp *.cxx)

foreach(sample_src ${srcs})
  get_filename_component(sample ${sample_src} NAME_WE)
  add_executable(${sample} ${sample_src})
  target_include_directories(${sample} PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${sample} PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endforeach()
