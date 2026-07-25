cmake_minimum_required(VERSION 3.16)
project(OpenCVColorization LANGUAGES CXX)

find_package(
    OpenCV REQUIRED
    COMPONENTS core dnn highgui imgcodecs imgproc videoio
)
if(OpenCV_VERSION VERSION_LESS 4.14)
    message(FATAL_ERROR "OpenCV 4.14 or newer is required")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(colorize_image colorizeImage.cpp)
add_executable(colorize_video colorizeVideo.cpp)
foreach(target IN ITEMS colorize_image colorize_video)
    target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(
            ${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
        )
    endif()
endforeach()

include(CTest)
if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/colorization_eccv16.onnx")
    add_test(
        NAME colorization_image_validation
        COMMAND colorize_image
            --input "${CMAKE_CURRENT_SOURCE_DIR}/greyscaleImage.png"
            --model "${CMAKE_CURRENT_SOURCE_DIR}/models/colorization_eccv16.onnx"
            --output "${CMAKE_CURRENT_BINARY_DIR}/colorized-image.png"
            --no-display
            --validate
    )
    add_test(
        NAME colorization_video_validation
        COMMAND colorize_video
            --input "${CMAKE_CURRENT_SOURCE_DIR}/greyscaleVideo.mp4"
            --model "${CMAKE_CURRENT_SOURCE_DIR}/models/colorization_eccv16.onnx"
            --output "${CMAKE_CURRENT_BINARY_DIR}/colorized-video.avi"
            --max-frames 2
            --no-display
            --validate
    )
endif()
