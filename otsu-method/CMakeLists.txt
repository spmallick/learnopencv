cmake_minimum_required(VERSION 3.16)

project(otsu_thresholding LANGUAGES CXX)

include(CTest)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_library(otsu_options INTERFACE)
target_compile_features(otsu_options INTERFACE cxx_std_17)
target_include_directories(otsu_options INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(
    otsu_options
    INTERFACE OTSU_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

foreach(target_name IN ITEMS otsu_implementation otsu_method)
    add_executable("${target_name}" "${target_name}.cpp")
    target_link_libraries("${target_name}" PRIVATE otsu_options ${OpenCV_LIBS})
endforeach()

if(BUILD_TESTING)
    add_executable(test_otsu tests/test_otsu.cpp)
    target_link_libraries(test_otsu PRIVATE otsu_options ${OpenCV_LIBS})
    add_test(NAME otsu_cpp_regression COMMAND test_otsu)
endif()
