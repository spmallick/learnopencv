cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(TRT_LIB_DIR ${CMAKE_BINARY_DIR})
set(TRT_OUT_DIR ${CMAKE_BINARY_DIR})

set(TRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}" CACHE STRING "TensorRT project version")
set(TRT_SOVERSION "${TRT_SO_MAJOR}" CACHE STRING "TensorRT library so version")
message("Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}")

set(CMAKE_SKIP_BUILD_RPATH True)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies

set(DEFAULT_CUDA_VERSION 10.2)
set(DEFAULT_CUDNN_VERSION 8.0)
set(DEFAULT_PROTOBUF_VERSION 3.0.0)
set(DEFAULT_CUB_VERSION 1.8.0)

# Dependency Version Resolution
set(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
message(STATUS "CUDA version set to ${CUDA_VERSION}")
set(CUDNN_VERSION ${DEFAULT_CUDNN_VERSION})
message(STATUS "cuDNN version set to ${CUDNN_VERSION}")

find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
)

find_library(CUDNN_LIB cudnn HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)

find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
find_library(RT_LIB rt)

set(CUDA_LIBRARIES ${CUDART_LIB})

set(GPU_ARCHS 72)

# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

add_executable(trt_test inferutils.cpp sample.cpp)
target_link_libraries(trt_test ${CUDA_LIBRARIES} nvinfer nvonnxparser stdc++fs)
