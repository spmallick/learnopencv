cmake_minimum_required(VERSION 3.16)
project(object_tracking_opencv LANGUAGES CXX)

find_package(OpenCV REQUIRED
    COMPONENTS core highgui imgcodecs imgproc video videoio
    OPTIONAL_COMPONENTS tracking
)

add_executable(object_tracker tracker.cpp)
target_compile_features(object_tracker PRIVATE cxx_std_17)
target_link_libraries(object_tracker PRIVATE ${OpenCV_LIBS})
target_compile_options(object_tracker PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME tracking_smoke
        COMMAND object_tracker
            --input=${CMAKE_CURRENT_SOURCE_DIR}/videos/chaplin.mp4
            --tracker=MIL
            --bbox=287,23,86,320
            --max-frames=5
            --snapshot=${CMAKE_CURRENT_BINARY_DIR}/tracking-smoke.png
    )
endif()
