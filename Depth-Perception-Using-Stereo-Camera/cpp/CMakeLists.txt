cmake_minimum_required(VERSION 3.16)

project(stereo_depth_opencv LANGUAGES CXX)

include(CTest)

find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_GREATER_EQUAL "5.0")
    set(OPENCV_COMPONENTS
        core imgproc imgcodecs highgui videoio stereo geometry
    )
else()
    set(OPENCV_COMPONENTS
        core imgproc imgcodecs highgui videoio calib3d
    )
endif()
find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_COMPONENTS})

set(OPENCV_TARGETS)
foreach(component IN LISTS OPENCV_COMPONENTS)
    list(APPEND OPENCV_TARGETS "opencv_${component}")
endforeach()

add_library(stereo_depth STATIC stereo_depth.cpp)
target_compile_features(stereo_depth PUBLIC cxx_std_17)
target_include_directories(stereo_depth PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(stereo_depth PUBLIC ${OPENCV_TARGETS})
target_compile_definitions(
    stereo_depth
    PUBLIC STEREO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
)

foreach(
    target_name
    IN ITEMS
        disparity_params_gui
        disparity2depth_calib
        obstacle_avoidance
        stereo_depth_cli
)
    add_executable("${target_name}" "${target_name}.cpp")
    target_link_libraries("${target_name}" PRIVATE stereo_depth)
endforeach()

if(BUILD_TESTING)
    add_executable(test_stereo_depth tests/test_stereo_depth.cpp)
    target_link_libraries(test_stereo_depth PRIVATE stereo_depth)
    add_test(NAME stereo_depth_cpp_regression COMMAND test_stereo_depth)
endif()
