cmake_minimum_required(VERSION 3.16)
project(blob_detector_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs)
if(OpenCV_VERSION VERSION_LESS 4.10 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
    message(FATAL_ERROR
        "This example supports OpenCV 4.10 through 5.x, not ${OpenCV_VERSION}.")
endif()

if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
    find_package(OpenCV REQUIRED COMPONENTS core features highgui imgcodecs)
else()
    find_package(OpenCV REQUIRED COMPONENTS core features2d highgui imgcodecs)
endif()

add_executable(blob_detector blob.cpp)
target_include_directories(blob_detector SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(blob_detector PRIVATE ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME blob_detector_sample
        COMMAND
            blob_detector
            --input ${CMAKE_CURRENT_SOURCE_DIR}/blob.jpg
            --output ${CMAKE_CURRENT_BINARY_DIR}/blob-keypoints-test.png
    )
    set_tests_properties(
        blob_detector_sample
        PROPERTIES PASS_REGULAR_EXPRESSION "Detected 16 blobs"
    )
endif()
