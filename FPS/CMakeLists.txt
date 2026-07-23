# Require modern target-scoped CMake behavior on every supported platform.
cmake_minimum_required(VERSION 3.16)

# Build the tutorial as a C++-only project.
project(frame_rate LANGUAGES CXX)

# Request only the OpenCV modules used by frame_rate.cpp.
find_package(OpenCV REQUIRED COMPONENTS core videoio)

# Keep the declared compatibility contract limited to OpenCV 4.x and 5.x.
if(OpenCV_VERSION VERSION_LESS 4 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  # Stop configuration before an unsupported API version can be compiled.
  message(FATAL_ERROR
    "FPS supports OpenCV 4.x and 5.x, not ${OpenCV_VERSION}")
endif()

# Record the resolved version in configure logs for validation evidence.
message(STATUS "FPS will build with OpenCV ${OpenCV_VERSION}")

# Compile the one shipped C++ example.
add_executable(frame_rate frame_rate.cpp)

# OpenCV 5 requires C++17, which is also supported by the OpenCV 4.14 target.
target_compile_features(frame_rate PRIVATE cxx_std_17)

# Disable compiler-specific language extensions for portable C++17 behavior.
set_target_properties(frame_rate PROPERTIES CXX_EXTENSIONS OFF)

# Apply OpenCV headers only to the tutorial executable.
target_include_directories(frame_rate PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link the requested modules and any transitive dependencies resolved by OpenCV.
target_link_libraries(frame_rate PRIVATE ${OpenCV_LIBS})

# Treat every enabled warning as an error in the acceptance build.
if(MSVC)
  # Use Microsoft's high warning level and fail on every warning.
  target_compile_options(frame_rate PRIVATE /W4 /WX)
else()
  # Use the common GCC/Clang strict-warning set and fail on every warning.
  target_compile_options(frame_rate PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Provide BUILD_TESTING and CTest integration for the real CLI regression.
include(CTest)

# Avoid imposing Python test dependencies on users who configure with testing off.
if(BUILD_TESTING)
  # The regression harness uses the same supported Python baseline as frame_rate.py.
  find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)

  # Run both real implementations against one generated video from an unrelated cwd.
  add_test(
    NAME frame_rate_regression
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_implementations.py"
      --python-script "${CMAKE_CURRENT_SOURCE_DIR}/frame_rate.py"
      --cpp-executable "$<TARGET_FILE:frame_rate>"
      --expected-opencv-version "${OpenCV_VERSION}"
  )

  # Require the final cross-language marker and bound accidental camera/test hangs.
  set_tests_properties(
    frame_rate_regression
    PROPERTIES
      PASS_REGULAR_EXPRESSION "Cross-language FPS validation passed"
      TIMEOUT 30
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
endif()
