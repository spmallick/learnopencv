cmake_minimum_required(VERSION 3.16)
project(homography_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)
if(OpenCV_VERSION VERSION_LESS 4.10 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
    message(FATAL_ERROR
        "These examples support OpenCV 4.10 through 5.x, not ${OpenCV_VERSION}.")
endif()

if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
    find_package(
        OpenCV REQUIRED
        COMPONENTS core geometry highgui imgcodecs imgproc
    )
else()
    find_package(
        OpenCV REQUIRED
        COMPONENTS calib3d core highgui imgcodecs imgproc
    )
endif()

set(EXAMPLE_TARGETS
    homography
    homography2
    perspective_correction
    virtual_billboard
)

add_executable(homography homography.cpp)
add_executable(homography2 homography2.cpp)
add_executable(perspective_correction perspective-correction.cpp)
add_executable(virtual_billboard virtual-billboard.cpp)

foreach(target IN LISTS EXAMPLE_TARGETS)
    target_include_directories(
        ${target} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(
        ${target}
        PRIVATE HOMOGRAPHY_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endforeach()

include(CTest)
if(BUILD_TESTING)
    set(BOOK_POINTS "318,256;534,372;316,670;73,473")
    set(BILLBOARD_POINTS "80,165;320,255;255,390;35,305")

    add_executable(
        homography_unit_tests tests/test_homography.cpp
    )
    target_include_directories(
        homography_unit_tests SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(homography_unit_tests PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(
        homography_unit_tests
        PRIVATE HOMOGRAPHY_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(NAME homography_unit_tests COMMAND homography_unit_tests)

    foreach(target IN ITEMS homography perspective_correction)
        add_test(
            NAME ${target}_sample
            COMMAND
                ${target}
                --input ${CMAKE_CURRENT_SOURCE_DIR}/book1.jpg
                --points "${BOOK_POINTS}"
                --output ${CMAKE_CURRENT_BINARY_DIR}/${target}-test.jpg
        )
        set_tests_properties(
            ${target}_sample
            PROPERTIES PASS_REGULAR_EXPRESSION "Saved 300x400"
        )
    endforeach()

    foreach(target IN ITEMS homography2 virtual_billboard)
        add_test(
            NAME ${target}_sample
            COMMAND
                ${target}
                --source ${CMAKE_CURRENT_SOURCE_DIR}/first-image.jpg
                --destination ${CMAKE_CURRENT_SOURCE_DIR}/times-square.jpg
                --points "${BILLBOARD_POINTS}"
                --output ${CMAKE_CURRENT_BINARY_DIR}/${target}-test.jpg
        )
        set_tests_properties(
            ${target}_sample
            PROPERTIES PASS_REGULAR_EXPRESSION "Saved composite image"
        )
    endforeach()
endif()
