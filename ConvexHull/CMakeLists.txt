cmake_minimum_required(VERSION 3.16)
project(ConvexHullExample LANGUAGES CXX)

# Establish the installed major version using module names shared by OpenCV 4
# and 5. Do not request a version here: OpenCV's config-version logic can reject
# a different major before the version-aware component selection below.
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
if(OpenCV_VERSION VERSION_LESS 4.8 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.8 through 5.x, not ${OpenCV_VERSION}")
endif()

if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
  # OpenCV 5 moved convexHull and related contour geometry to geometry.
  find_package(
    OpenCV REQUIRED
    COMPONENTS core geometry highgui imgcodecs imgproc
  )
else()
  # OpenCV 4 declares and implements convexHull in imgproc.
  find_package(
    OpenCV REQUIRED
    COMPONENTS core highgui imgcodecs imgproc
  )
endif()

add_executable(ConvexHull example.cpp)
target_compile_features(ConvexHull PRIVATE cxx_std_17)
target_include_directories(ConvexHull PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ConvexHull PRIVATE ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME ConvexHullCppSmoke
    COMMAND
      ConvexHull
      --input=${CMAKE_CURRENT_SOURCE_DIR}/sample.jpg
      --output=${CMAKE_CURRENT_BINARY_DIR}/convex-hull-cpp-test.jpg
  )
  set_tests_properties(
    ConvexHullCppSmoke
    PROPERTIES PASS_REGULAR_EXPRESSION "Detected [1-9][0-9]* contours"
  )
endif()
