cmake_minimum_required(VERSION 3.0.0)
project(contour_approximations VERSION 0.1.0)

include(CTest)
enable_testing()


find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(contour_approximations contour_approx.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX__STANDARD_REQUIRED ON)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(contour_approximations ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})