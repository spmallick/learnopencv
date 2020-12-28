cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

PROJECT(stereo)

find_package( OpenCV REQUIRED )


MACRO(add_example name)
  ADD_EXECUTABLE(${name} ${name}.cpp)
  TARGET_LINK_LIBRARIES(${name} ${OpenCV_LIBS})
ENDMACRO()

add_example(stereo)
