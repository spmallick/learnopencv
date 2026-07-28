cmake_minimum_required(VERSION 3.16)
project(fill_holes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)
if(OpenCV_VERSION VERSION_LESS "4.8" OR NOT OpenCV_VERSION VERSION_LESS "6.0")
    message(FATAL_ERROR "This example supports OpenCV 4.8+ and OpenCV 5.x")
endif()

add_executable(imfill imfill.cpp)
target_link_libraries(imfill PRIVATE ${OpenCV_LIBS})
target_compile_options(
    imfill PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME imfill_cpp_smoke
        COMMAND imfill
            "${CMAKE_CURRENT_SOURCE_DIR}/nickel.jpg"
            --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output"
    )
endif()
