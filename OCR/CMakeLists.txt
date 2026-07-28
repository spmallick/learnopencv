cmake_minimum_required(VERSION 3.16)
project(tesseract_ocr_opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
if(OpenCV_VERSION VERSION_LESS "4.8" OR NOT OpenCV_VERSION VERSION_LESS "6.0")
    message(FATAL_ERROR "This example supports OpenCV 4.8+ and OpenCV 5.x")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(Tesseract REQUIRED IMPORTED_TARGET "tesseract>=5.0")

add_executable(ocr_simple ocr_simple.cpp)
target_link_libraries(
    ocr_simple PRIVATE
    ${OpenCV_LIBS}
    PkgConfig::Tesseract
)
target_compile_options(
    ocr_simple PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME ocr_cpp_smoke
        COMMAND ocr_simple
            "${CMAKE_CURRENT_SOURCE_DIR}/images/road-sign-3.jpg"
            --preprocess gray
            --psm 6
    )
    set_tests_properties(
        ocr_cpp_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "VIDEO SURVEILLANCE"
    )
endif()
