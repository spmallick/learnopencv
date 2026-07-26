cmake_minimum_required(VERSION 3.16)
project(SuperResolution LANGUAGES CXX)

# Discover without a requested version first. OpenCV's package-version file
# can reject a different major before this project gets a chance to select
# version-specific modules.
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
if(OpenCV_VERSION VERSION_LESS 4.8 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.8 through 5.x, not ${OpenCV_VERSION}")
endif()

# dnn_superres remains an opencv_contrib component with the same public header
# in both supported major versions. Request it explicitly so a core-only OpenCV
# installation fails during configuration with a useful missing-component error.
find_package(OpenCV REQUIRED COMPONENTS core dnn dnn_superres imgcodecs)

add_executable(SuperResolution super_res.cpp)
target_compile_features(SuperResolution PRIVATE cxx_std_17)
target_include_directories(SuperResolution PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(SuperResolution ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
  set(SUPERRES_TEST_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/models/ESPCN_x4.pb")
  if(EXISTS "${SUPERRES_TEST_MODEL}")
    add_test(
      NAME SuperResolutionCppInference
      COMMAND
        SuperResolution
        --input=${CMAKE_CURRENT_SOURCE_DIR}/image.png
        --model=${SUPERRES_TEST_MODEL}
        --algorithm=espcn
        --scale=4
        --output=${CMAKE_CURRENT_BINARY_DIR}/super-resolution-cpp-test.png
    )
    set_tests_properties(
      SuperResolutionCppInference
      PROPERTIES
        PASS_REGULAR_EXPRESSION
          "Upscaled 600x400 -> 2400x1600"
    )
  else()
    message(STATUS
      "ESPCN x4 model not found; run download_model.py before configuring "
      "to enable SuperResolutionCppInference")
  endif()
endif()
