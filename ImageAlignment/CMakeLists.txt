cmake_minimum_required(VERSION 3.16)
project(image_alignment_ecc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui video)

add_executable(image_alignment image_alignment.cpp)
target_link_libraries(image_alignment PRIVATE ${OpenCV_LIBS})

add_executable(image_alignment_simple image_alignment_simple_example.cpp)
target_link_libraries(image_alignment_simple PRIVATE ${OpenCV_LIBS})
