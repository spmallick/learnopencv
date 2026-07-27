cmake_minimum_required(VERSION 3.16)
project(lens_distortion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "5.0")
    set(OPENCV_COMPONENTS
        core imgproc imgcodecs highgui calib objdetect geometry
    )
else()
    set(OPENCV_COMPONENTS
        core imgproc imgcodecs highgui calib3d
    )
endif()
find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_COMPONENTS})

set(OPENCV_TARGETS)
foreach(component IN LISTS OPENCV_COMPONENTS)
    list(APPEND OPENCV_TARGETS "opencv_${component}")
endforeach()

add_library(lens_calibration lens_calibration.cpp)
target_include_directories(
    lens_calibration
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(lens_calibration PUBLIC ${OPENCV_TARGETS})

add_executable(Undistort Undistort.cpp)
target_link_libraries(Undistort PRIVATE lens_calibration)
target_compile_definitions(
    Undistort PRIVATE LENS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

include(CTest)
if(BUILD_TESTING)
    add_executable(
        lens_distortion_tests
        tests/test_lens_distortion.cpp
    )
    target_link_libraries(
        lens_distortion_tests PRIVATE lens_calibration
    )
    target_compile_definitions(
        lens_distortion_tests
        PRIVATE LENS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME lens_distortion_tests COMMAND lens_distortion_tests)
endif()
