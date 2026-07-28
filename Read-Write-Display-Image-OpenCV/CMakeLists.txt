cmake_minimum_required(VERSION 3.16)
project(image_io_opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)

add_executable(image_io image_io.cpp)
target_link_libraries(image_io PRIVATE ${OpenCV_LIBS})
target_include_directories(image_io PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MSVC)
  target_compile_options(image_io PRIVATE /W4 /WX)
else()
  target_compile_options(image_io PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME image_io_self_test COMMAND image_io --self-test)
endif()
