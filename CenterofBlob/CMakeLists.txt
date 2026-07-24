# OpenCV 5 requires C++17; CMake 3.16 provides dependable target-based support.
cmake_minimum_required(VERSION 3.16)

# Restrict this tutorial project to the one language it actually builds.
project(center_of_blob LANGUAGES CXX)

# Make the language contract explicit instead of relying on compiler defaults.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Request only the shared modules plus geometry as an optional split module.
# OpenCV 4 does not provide geometry; OpenCV 5 must provide it and is checked
# explicitly below after the single package lookup reports the resolved version.
find_package(
    OpenCV CONFIG REQUIRED
    COMPONENTS core highgui imgcodecs imgproc
    OPTIONAL_COMPONENTS geometry)

# This migration intentionally supports OpenCV major versions 4 and 5 only.
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
    message(FATAL_ERROR
        "CenterofBlob supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}")
endif()

# OpenCV 5 moved moments and contourArea from imgproc into geometry.
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5 AND NOT OpenCV_geometry_FOUND)
    message(FATAL_ERROR
        "OpenCV 5 was found without the required geometry module")
endif()

# Report the resolved version so CI logs prove which matrix cell was built.
message(STATUS "CenterofBlob OpenCV version: ${OpenCV_VERSION}")

# Build the two canonical examples and the historical misspelled compatibility
# source that remains part of the published project.
foreach(example
        IN ITEMS
        single_blob
        center_of_multiple_blob
        center_of_multipe_blob)
    # Each source is a complete executable tutorial entry point.
    add_executable(${example} ${example}.cpp)

    # Keep include paths, libraries, and source location scoped to the target.
    target_include_directories(${example} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${example} PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(
        ${example}
        PRIVATE CENTER_OF_BLOB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    # Treat warnings as errors so both migration paths remain clean.
    if(MSVC)
        target_compile_options(${example} PRIVATE /W4 /WX)
    else()
        target_compile_options(
            ${example} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endforeach()

# Enable CTest without forcing downstream projects to run tests.
include(CTest)

# Respect -DBUILD_TESTING=OFF when this tutorial is embedded in another build.
if(BUILD_TESTING)
    # Register a clean, headless invocation with an exact one-file output contract.
    function(add_blob_example_test test_name target_name expected_output marker)
        # Isolate each test's artifact so stale files cannot hide a regression.
        set(output_directory
            "${CMAKE_CURRENT_BINARY_DIR}/test-output/${test_name}")

        # The script removes only this build-local directory, invokes the real
        # executable, requires its marker, and checks the exact output set.
        add_test(
            NAME ${test_name}
            COMMAND
                ${CMAKE_COMMAND}
                "-DEXECUTABLE=$<TARGET_FILE:${target_name}>"
                "-DOUTPUT_DIR=${output_directory}"
                "-DEXPECTED_OUTPUT=${expected_output}"
                "-DEXPECTED_MARKER=${marker}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunExampleTest.cmake")
    endfunction()

    # Test the single-blob entry point against its bundled-input default.
    add_blob_example_test(
        single_blob_bundled_image
        single_blob
        single_blob_result.png
        "VALIDATION PASSED: single_blob")

    # Test the canonical multiple-blob entry point and its six known contours.
    add_blob_example_test(
        multiple_blob_bundled_image
        center_of_multiple_blob
        multiple_blob_result.png
        "VALIDATION PASSED: center_of_multiple_blob")

    # Compile and run the misspelled filename so published legacy links work.
    add_blob_example_test(
        multiple_blob_legacy_filename
        center_of_multipe_blob
        multiple_blob_result.png
        "VALIDATION PASSED: center_of_multiple_blob")
endif()
