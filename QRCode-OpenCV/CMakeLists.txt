cmake_minimum_required(VERSION 3.16)
project(qr_code_opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
  OpenCV REQUIRED
  COMPONENTS core highgui imgcodecs imgproc objdetect
)
if(OpenCV_VERSION VERSION_LESS 4.14 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}"
  )
endif()

add_executable(qrCodeOpencv qrCodeOpencv.cpp)
target_compile_definitions(
  qrCodeOpencv PRIVATE QR_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_include_directories(qrCodeOpencv PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(qrCodeOpencv PRIVATE ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(qrCodeOpencv PRIVATE /W4 /WX /permissive-)
else()
  target_compile_options(
    qrCodeOpencv PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME qr_code_regression
    COMMAND qrCodeOpencv
      --no-display
      --validate
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test-output"
  )
  set_tests_properties(
    qr_code_regression
    PROPERTIES PASS_REGULAR_EXPRESSION "VALIDATION PASSED"
  )
endif()
