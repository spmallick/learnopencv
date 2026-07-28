cmake_minimum_required(VERSION 3.16)
project(video_read_write_display LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS core highgui videoio)

add_executable(video_read videoRead.cpp)
add_executable(video_write videoWrite.cpp)
target_compile_features(video_read PRIVATE cxx_std_17)
target_compile_features(video_write PRIVATE cxx_std_17)
target_link_libraries(video_read PRIVATE ${OpenCV_LIBS})
target_link_libraries(video_write PRIVATE ${OpenCV_LIBS})

foreach(target video_read video_write)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
    )
endforeach()

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME video_read_smoke
        COMMAND video_read
            --input=${CMAKE_CURRENT_SOURCE_DIR}/chaplin.mp4
            --max-frames=8
    )
    add_test(
        NAME video_write_smoke
        COMMAND video_write
            --input=${CMAKE_CURRENT_SOURCE_DIR}/chaplin.mp4
            --output=${CMAKE_CURRENT_BINARY_DIR}/video-smoke.avi
            --codec=MJPG
            --max-frames=8
    )
    set_tests_properties(video_write_smoke PROPERTIES
        DEPENDS video_read_smoke
    )
endif()
