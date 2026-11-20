# The official Microsoft NuGet package includes matching C++ headers and native
# CPU libraries for both tested platforms. Nothing here requires Python.
set(ONNXRUNTIME_ROOT "" CACHE PATH "Existing ONNX Runtime installation")
set(ROBUST_ALIGNMENT_DOWNLOAD_ONNXRUNTIME ON CACHE BOOL "Download pinned native ONNX Runtime when no installation is supplied")

if(NOT ONNXRUNTIME_ROOT)
  if(NOT ROBUST_ALIGNMENT_DOWNLOAD_ONNXRUNTIME)
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT or enable ROBUST_ALIGNMENT_DOWNLOAD_ONNXRUNTIME")
  endif()
  set(_ort_version "1.24.3")
  set(_ort_cache "${CMAKE_BINARY_DIR}/_deps/onnxruntime-${_ort_version}")
  set(_ort_package "${CMAKE_BINARY_DIR}/_deps/microsoft.ml.onnxruntime.${_ort_version}.nupkg")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_ort_platform "osx-arm64")
    set(_ort_filename "libonnxruntime.dylib")
  elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_ort_platform "osx-x64")
    set(_ort_filename "libonnxruntime.dylib")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(_ort_platform "linux-x64")
    set(_ort_filename "libonnxruntime.so")
  else()
    message(FATAL_ERROR "Provide ONNXRUNTIME_ROOT for this platform")
  endif()
  if(NOT EXISTS "${_ort_cache}/runtimes/${_ort_platform}/native/${_ort_filename}")
    file(DOWNLOAD
      "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${_ort_version}/microsoft.ml.onnxruntime.${_ort_version}.nupkg"
      "${_ort_package}"
      EXPECTED_HASH SHA256=e4ab3b236c1b803e1e7167fe9a75caa1b82a2fd4ae82abbd130937c07517bcdb
      TLS_VERIFY ON SHOW_PROGRESS STATUS _ort_status)
    list(GET _ort_status 0 _ort_code)
    if(NOT _ort_code EQUAL 0)
      message(FATAL_ERROR "Unable to download pinned ONNX Runtime: ${_ort_status}")
    endif()
    file(MAKE_DIRECTORY "${_ort_cache}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xvf "${_ort_package}"
      "build/native/include/*" "runtimes/${_ort_platform}/native/*"
      "LICENSE" "ThirdPartyNotices.txt" "Privacy.md"
      WORKING_DIRECTORY "${_ort_cache}" RESULT_VARIABLE _ort_extract OUTPUT_QUIET)
    if(NOT _ort_extract EQUAL 0)
      message(FATAL_ERROR "Unable to extract ONNX Runtime package")
    endif()
  endif()
  set(ONNXRUNTIME_INCLUDE_DIR "${_ort_cache}/build/native/include")
  set(ONNXRUNTIME_LIBRARY "${_ort_cache}/runtimes/${_ort_platform}/native/${_ort_filename}")
  # NuGet uses an unversioned filename although each native library embeds a
  # versioned loader name. Supply that name beside it for the system loader.
  get_filename_component(_ort_native_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  if(APPLE)
    file(CREATE_LINK "${_ort_filename}" "${_ort_native_dir}/libonnxruntime.${_ort_version}.dylib" SYMBOLIC)
  else()
    file(CREATE_LINK "${_ort_filename}" "${_ort_native_dir}/libonnxruntime.so.1" SYMBOLIC)
  endif()
else()
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/build/native/include" REQUIRED NO_DEFAULT_PATH)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" REQUIRED NO_DEFAULT_PATH)
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR "ONNX Runtime native library missing: ${ONNXRUNTIME_LIBRARY}")
endif()
add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED GLOBAL)
set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
