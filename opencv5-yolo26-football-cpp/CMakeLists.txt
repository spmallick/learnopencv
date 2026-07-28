cmake_minimum_required(VERSION 3.16)
project(opencv5_yolo26_football_cpp LANGUAGES CXX)

# OpenCV 5 + YOLO26 football demos in C++ (Part 2 companion code).
# Detection, team split, pose estimation and instance segmentation, all through
# the OpenCV 5 DNN module on the CPU.
#
# Build (pointing at an OpenCV 5 install; see build_opencv5.sh):
#   cmake -G Ninja -B build -DOpenCV_DIR=/path/to/opencv5_install/lib/cmake/opencv5
#   cmake --build build

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV 5 REQUIRED COMPONENTS core dnn imgcodecs imgproc videoio)
if(OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR "This project supports OpenCV 5.x, not ${OpenCV_VERSION}")
endif()
find_package(Threads REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_library(yolo26_dnn STATIC src/yolo26_dnn.cpp)
target_include_directories(yolo26_dnn PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo26_dnn PUBLIC ${OpenCV_LIBS})

function(enable_strict_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

enable_strict_warnings(yolo26_dnn)

set(DEMOS
  detect_image        # single-image object detection
  detect_video        # video object detection + FPS
  detect_teams        # split players into teams by shirt colour
  detect_pose         # YOLO26-pose keypoints / skeleton
  detect_seg          # YOLO26-seg instance masks
  benchmark_engines   # new vs classic DNN engine
)
foreach(demo ${DEMOS})
  add_executable(${demo} src/${demo}.cpp)
  target_link_libraries(${demo} PRIVATE yolo26_dnn ${OpenCV_LIBS} Threads::Threads)
  enable_strict_warnings(${demo})
endforeach()
