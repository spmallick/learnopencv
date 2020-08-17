cmake_minimum_required(VERSION 3.5.1)

project(ocr_simple)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

find_package( PkgConfig REQUIRED)
pkg_search_module(TESSERACT REQUIRED tesseract)

add_executable(ocr_simple ocr_simple.cpp)
target_include_directories(ocr_simple PRIVATE ${OpenCV_INCLUDE_DIRS} ${TESSERACT_INCLUDE_DIRS})
target_link_directories(ocr_simple PRIVATE ${TESSERACT_LIBRARY_DIRS})
target_link_libraries(ocr_simple PRIVATE ${OpenCV_LIBS} ${TESSERACT_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
