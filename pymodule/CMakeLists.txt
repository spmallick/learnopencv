cmake_minimum_required(VERSION 3.12)

project(sample)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

set(PythonGeneratedIncludes
    "${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude/pybv_signatures.json"
    "${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude/pybv_generated_funcs.h"
    "${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude/pybv_generated_include.h"
    "${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude/pybv_generated_ns_reg.h"
    "${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude/pybv_generated_type_publish.h"
    "${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude/pybv_generated_type_reg.h"
    "${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude/pybv_generated_types.h"
)

add_custom_command(
    OUTPUT ${PythonGeneratedIncludes}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gen2.py pybv ${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude ${CMAKE_CURRENT_LIST_DIR}/headers.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/headers.txt ${CMAKE_CURRENT_LIST_DIR}/gen2.py
    COMMENT "Generating C header files for Python interface"
    VERBATIM
)

add_custom_target(python_generated_headers DEPENDS "${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude/pybv_signatures.json")

Python_add_library(bv MODULE bv.cpp src/bvmodule.cpp)
target_include_directories(bv PRIVATE "${CMAKE_BINARY_DIR}/PythonPackageGeneratedInclude" "${OpenCV_INCLUDE_DIRS}" "${CMAKE_CURRENT_LIST_DIR}")
target_compile_definitions(bv PRIVATE -DMODULE_STR=bv -DMODULE_PREFIX=pybv)
target_link_libraries(bv PRIVATE "${OpenCV_LIBS}" Python::NumPy "${CMAKE_THREAD_LIBS_INIT}")
add_dependencies(bv python_generated_headers)
