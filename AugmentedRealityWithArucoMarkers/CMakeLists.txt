cmake_minimum_required(VERSION 3.16)
project(AugmentedRealityWithArucoMarkers LANGUAGES CXX)

# Establish the OpenCV major using module names shared by OpenCV 4 and 5.
# Requesting 4.8 directly can make OpenCV's package-version logic reject 5.x
# before the version-aware ArUco component selection below.
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
if(OpenCV_VERSION VERSION_LESS 4.8 OR OpenCV_VERSION VERSION_GREATER_EQUAL 6)
  message(FATAL_ERROR
    "This example supports OpenCV 4.8 through 5.x, not ${OpenCV_VERSION}")
endif()

if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
  # OpenCV 5 keeps ArUco under objdetect.
  find_package(
    OpenCV REQUIRED
    COMPONENTS core geometry highgui imgcodecs imgproc objdetect videoio
  )
else()
  # OpenCV 4.x exposes the modern objdetect/aruco headers, while the compiled
  # implementation is still delivered by the separate aruco component.
  find_package(
    OpenCV REQUIRED
    COMPONENTS aruco core highgui imgcodecs imgproc videoio
  )
endif()

add_executable(
  AugmentedRealityWithAruco
  augmented_reality_with_aruco.cpp
)
target_compile_features(AugmentedRealityWithAruco PRIVATE cxx_std_17)
target_include_directories(
  AugmentedRealityWithAruco
  PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(AugmentedRealityWithAruco PRIVATE ${OpenCV_LIBS})

add_executable(GenerateArucoMarker generate_aruco_markers.cpp)
target_compile_features(GenerateArucoMarker PRIVATE cxx_std_17)
target_include_directories(GenerateArucoMarker PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(GenerateArucoMarker PRIVATE ${OpenCV_LIBS})

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME ArucoCppImageSmoke
    COMMAND
      AugmentedRealityWithAruco
      --image=${CMAKE_CURRENT_SOURCE_DIR}/test.jpg
      --overlay=${CMAKE_CURRENT_SOURCE_DIR}/new_scenery.jpg
      --output=${CMAKE_CURRENT_BINARY_DIR}/aruco-cpp-test.jpg
      --augmented-only=true
      --strict=true
  )
  set_tests_properties(
    ArucoCppImageSmoke
    PROPERTIES PASS_REGULAR_EXPRESSION "augmented; detected IDs 23,25,30,33"
  )

  add_test(
    NAME ArucoCppMarkerSmoke
    COMMAND
      GenerateArucoMarker
      --id=33
      --size=200
      --output=${CMAKE_CURRENT_BINARY_DIR}/marker33-cpp-test.png
  )
  set_tests_properties(
    ArucoCppMarkerSmoke
    PROPERTIES PASS_REGULAR_EXPRESSION "Wrote marker 33"
  )
endif()
