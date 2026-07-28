cmake_minimum_required(VERSION 3.16)
project(image_cropping_opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)

add_executable(crop_image crop_image.cpp)
target_link_libraries(crop_image PRIVATE ${OpenCV_LIBS})
target_include_directories(crop_image PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MSVC)
  target_compile_options(crop_image PRIVATE /W4 /WX)
else()
  target_compile_options(crop_image PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME crop_image_self_test COMMAND crop_image --self-test)
endif()
