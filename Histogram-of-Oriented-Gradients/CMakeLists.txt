cmake_minimum_required(VERSION 3.16)
project(histogram_of_oriented_gradients LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(hog_demo hog.cpp)
target_compile_features(hog_demo PRIVATE cxx_std_17)
target_link_libraries(hog_demo PRIVATE ${OpenCV_LIBS})
target_compile_options(hog_demo PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME hog_smoke
        COMMAND hog_demo
            --output-dir=${CMAKE_CURRENT_BINARY_DIR}/hog-output
    )
endif()
