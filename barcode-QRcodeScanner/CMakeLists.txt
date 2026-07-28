cmake_minimum_required(VERSION 3.16)
project(barcode_qr_scanner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    OpenCV REQUIRED
    COMPONENTS core highgui imgcodecs imgproc objdetect
)
if(OpenCV_VERSION VERSION_LESS "4.8" OR NOT OpenCV_VERSION VERSION_LESS "6.0")
    message(FATAL_ERROR "This example supports OpenCV 4.8+ and OpenCV 5.x")
endif()

add_executable(barcode_qr_scanner barcode-QRcodeScanner.cpp)
target_link_libraries(barcode_qr_scanner PRIVATE ${OpenCV_LIBS})
target_compile_options(
    barcode_qr_scanner PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME barcode_qr_cpp_qr_smoke
        COMMAND barcode_qr_scanner
            "${CMAKE_CURRENT_SOURCE_DIR}/zbar-test.jpg"
            --no-barcode
            --output "${CMAKE_CURRENT_BINARY_DIR}/qr-smoke.png"
    )
    set_tests_properties(
        barcode_qr_cpp_qr_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "QR_CODE[\t]http://LearnOpenCV.com"
    )
endif()
