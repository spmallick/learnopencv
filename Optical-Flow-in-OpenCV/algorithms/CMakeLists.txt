cmake_minimum_required(VERSION 3.16)
project(optical_flow_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS 4.14 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}"
  )
endif()
if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
  find_package(
    OpenCV REQUIRED
    COMPONENTS core features highgui imgcodecs imgproc optflow video videoio
  )
else()
  find_package(
    OpenCV REQUIRED
    COMPONENTS core highgui imgcodecs imgproc optflow video videoio
  )
endif()

add_executable(
  OpticalFlow
  main.cpp
  lucas-kanade.cpp
  dense_optical_flow.cpp
)
target_compile_definitions(
  OpticalFlow
  PRIVATE OPTICAL_FLOW_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_include_directories(
  OpticalFlow PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(OpticalFlow PRIVATE ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(OpticalFlow PRIVATE /W4 /WX /permissive-)
else()
  target_compile_options(
    OpticalFlow PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()

include(CTest)
if(BUILD_TESTING)
  foreach(
    algorithm
    farneback
    lucaskanade
    lucaskanade_dense
    rlof
  )
    add_test(
      NAME "optical_flow_${algorithm}"
      COMMAND OpticalFlow
        --algorithm "${algorithm}"
        --no-display
        --validate
        --max-frames 3
        --output-dir
          "${CMAKE_CURRENT_BINARY_DIR}/test-output/${algorithm}"
    )
    set_tests_properties(
      "optical_flow_${algorithm}"
      PROPERTIES PASS_REGULAR_EXPRESSION "VALIDATION PASSED"
    )
  endforeach()
endif()
