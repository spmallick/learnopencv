# Run one tracker regression and independently verify its observable contract.
# Keeping these checks in a CMake script makes them available anywhere CTest
# runs, without requiring a particular shell, Python interpreter, or codec tool.

cmake_minimum_required(VERSION 3.16)

# Stop immediately when the test registration omitted a required value. A
# configuration typo should not be mistaken for a tracker regression.
foreach(_required_variable IN ITEMS EXECUTABLE TRACKER OUTPUT_DIR)
  if(NOT DEFINED ${_required_variable} OR "${${_required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${_required_variable} must be provided")
  endif()
endforeach()

# Every run starts from an empty directory so stale files cannot satisfy the
# artifact checks and unexpected new outputs are visible in the manifest.
file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

execute_process(
  COMMAND "${EXECUTABLE}"
          "--tracker=${TRACKER}"
          --validate
          --no-display
          "--output-dir=${OUTPUT_DIR}"
  RESULT_VARIABLE _validation_result
  OUTPUT_VARIABLE _validation_stdout
  ERROR_VARIABLE _validation_stderr)
set(_validation_log "${_validation_stdout}${_validation_stderr}")

# A successful process must also print the tutorial's explicit validation
# marker; checking both prevents a swallowed failure or accidental early exit.
if(NOT "${_validation_result}" STREQUAL "0")
  message(FATAL_ERROR
    "${TRACKER} validation exited ${_validation_result}:\n${_validation_log}")
endif()
if(NOT _validation_log MATCHES "VALIDATION PASSED")
  message(FATAL_ERROR
    "${TRACKER} validation omitted VALIDATION PASSED:\n${_validation_log}")
endif()

# Validation promises exactly these three files. Sort both lists before
# comparing so the check is independent of filesystem enumeration order.
set(_expected_artifacts
    "metrics_${TRACKER}.json"
    "synthetic_clip.avi"
    "tracked_${TRACKER}.avi")
list(SORT _expected_artifacts)
file(GLOB _actual_artifacts
     LIST_DIRECTORIES false
     RELATIVE "${OUTPUT_DIR}"
     "${OUTPUT_DIR}/*")
list(SORT _actual_artifacts)
if(NOT "${_actual_artifacts}" STREQUAL "${_expected_artifacts}")
  message(FATAL_ERROR
    "${TRACKER} artifacts differ.\n"
    "Expected: ${_expected_artifacts}\n"
    "Actual:   ${_actual_artifacts}")
endif()

# Empty media or JSON files are never useful even if their names are correct.
foreach(_artifact IN LISTS _expected_artifacts)
  set(_artifact_path "${OUTPUT_DIR}/${_artifact}")
  file(SIZE "${_artifact_path}" _artifact_size)
  if(_artifact_size EQUAL 0)
    message(FATAL_ERROR "${_artifact_path} is empty")
  endif()
endforeach()

# Require every documented metrics field. Quoted-key matching avoids accepting
# a field name merely because the same text appeared inside a string value.
set(_metrics_path "${OUTPUT_DIR}/metrics_${TRACKER}.json")
file(READ "${_metrics_path}" _metrics_json)
foreach(_metrics_key IN ITEMS
        tracker opencv_version frames lost_frames mean_fps mean_iou success_rate)
  string(FIND "${_metrics_json}" "\"${_metrics_key}\"" _key_position)
  if(_key_position EQUAL -1)
    message(FATAL_ERROR
      "${_metrics_path} is missing the '${_metrics_key}' key")
  endif()
endforeach()
if(NOT _metrics_json MATCHES "\"frames\"[ \t]*:[ \t]*80([,\\n])")
  message(FATAL_ERROR
    "${_metrics_path} does not report the expected 80 processed frames")
endif()
if(NOT _metrics_json MATCHES "\"lost_frames\"[ \t]*:[ \t]*0([,\\n])")
  message(FATAL_ERROR
    "${_metrics_path} reports a lost frame on the deterministic clip")
endif()

# Open both AVIs through the real example entry point. MIL is model-free and
# processing to end-of-stream proves that both files decode to all 80 expected
# frames. This catches the historical bug where tracked output omitted the
# initialization frame even though metrics reported it.
foreach(_video_name IN ITEMS synthetic_clip.avi tracked_${TRACKER}.avi)
  set(_video_path "${OUTPUT_DIR}/${_video_name}")
  execute_process(
    COMMAND "${EXECUTABLE}"
            --tracker=mil
            "--input=${_video_path}"
            --bbox=20,20,40,40
            --max-frames=81
            --no-display
    RESULT_VARIABLE _read_result
    OUTPUT_VARIABLE _read_stdout
    ERROR_VARIABLE _read_stderr)
  set(_read_log "${_read_stdout}${_read_stderr}")
  if(NOT "${_read_result}" STREQUAL "0")
    message(FATAL_ERROR
      "Could not read ${_video_path}:\n${_read_log}")
  endif()
  if(NOT _read_log MATCHES "frames=80([^0-9]|$)")
    message(FATAL_ERROR
      "Expected 80 decodable frames in ${_video_path}:\n${_read_log}")
  endif()
endforeach()

# MIL's validation test also covers the normal-mode JSON and one-frame output
# contract once per CTest run. Normal tracking has no ground truth, so both
# score fields must remain present as JSON null rather than disappearing.
if(TRACKER STREQUAL "mil")
  set(_normal_dir "${OUTPUT_DIR}_normal")
  file(REMOVE_RECURSE "${_normal_dir}")
  file(MAKE_DIRECTORY "${_normal_dir}")
  execute_process(
    COMMAND "${EXECUTABLE}"
            --tracker=mil
            "--input=${OUTPUT_DIR}/synthetic_clip.avi"
            --bbox=20,148,64,64
            --max-frames=1
            --no-display
            "--output-dir=${_normal_dir}"
    RESULT_VARIABLE _normal_result
    OUTPUT_VARIABLE _normal_stdout
    ERROR_VARIABLE _normal_stderr)
  set(_normal_log "${_normal_stdout}${_normal_stderr}")
  if(NOT "${_normal_result}" STREQUAL "0" OR
     NOT _normal_log MATCHES "frames=1([^0-9]|$)")
    message(FATAL_ERROR
      "Normal one-frame run failed its contract:\n${_normal_log}")
  endif()

  set(_normal_expected "metrics_mil.json" "tracked_mil.avi")
  list(SORT _normal_expected)
  file(GLOB _normal_actual
       LIST_DIRECTORIES false
       RELATIVE "${_normal_dir}"
       "${_normal_dir}/*")
  list(SORT _normal_actual)
  if(NOT "${_normal_actual}" STREQUAL "${_normal_expected}")
    message(FATAL_ERROR
      "Normal artifacts differ. Expected ${_normal_expected}; "
      "actual ${_normal_actual}")
  endif()
  foreach(_normal_artifact IN LISTS _normal_expected)
    file(SIZE "${_normal_dir}/${_normal_artifact}" _normal_size)
    if(_normal_size EQUAL 0)
      message(FATAL_ERROR
        "${_normal_dir}/${_normal_artifact} is empty")
    endif()
  endforeach()

  file(READ "${_normal_dir}/metrics_mil.json" _normal_metrics)
  if(NOT _normal_metrics MATCHES "\"frames\"[ \t]*:[ \t]*1([,\\n])" OR
     NOT _normal_metrics MATCHES "\"mean_iou\"[ \t]*:[ \t]*null" OR
     NOT _normal_metrics MATCHES "\"success_rate\"[ \t]*:[ \t]*null")
    message(FATAL_ERROR
      "Normal metrics schema or frame count is wrong:\n${_normal_metrics}")
  endif()

  # Decode the one-frame artifact through the real executable so a container
  # header without readable media cannot satisfy the size check above.
  execute_process(
    COMMAND "${EXECUTABLE}"
            --tracker=mil
            "--input=${_normal_dir}/tracked_mil.avi"
            --bbox=20,148,64,64
            --max-frames=2
            --no-display
    RESULT_VARIABLE _normal_read_result
    OUTPUT_VARIABLE _normal_read_stdout
    ERROR_VARIABLE _normal_read_stderr)
  set(_normal_read_log "${_normal_read_stdout}${_normal_read_stderr}")
  if(NOT "${_normal_read_result}" STREQUAL "0" OR
     NOT _normal_read_log MATCHES "frames=1([^0-9]|$)")
    message(FATAL_ERROR
      "Normal one-frame AVI was not readable:\n${_normal_read_log}")
  endif()
endif()

message(STATUS "${TRACKER}: validation outputs and readback checks passed")
