# Exercise the delivered executable without Python or a current-directory shortcut.
cmake_minimum_required(VERSION 3.18)

foreach(required EXECUTABLE PROJECT_DIR OUTPUT_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "check_cli.cmake requires -D${required}=...")
  endif()
endforeach()
if(NOT EXISTS "${EXECUTABLE}" OR IS_DIRECTORY "${EXECUTABLE}")
  message(FATAL_ERROR "Executable does not exist: ${EXECUTABLE}")
endif()

# A fresh child leaves previous test evidence and arbitrary caller-owned files intact.
get_filename_component(PROJECT_DIR "${PROJECT_DIR}" ABSOLUTE)
get_filename_component(OUTPUT_DIR "${OUTPUT_DIR}" ABSOLUTE)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef run_id)
set(run_dir "${OUTPUT_DIR}/contract-${run_id}")
set(working_dir "${run_dir}/unrelated-working-directory")
set(result_dir "${run_dir}/result")
file(MAKE_DIRECTORY "${working_dir}")

set(reference "${PROJECT_DIR}/assets/downloads/opencv/box.png")
set(scene "${PROJECT_DIR}/assets/downloads/opencv/box_in_scene.png")
set(negative "${PROJECT_DIR}/assets/downloads/opencv/fruits.jpg")
foreach(input "${reference}" "${scene}" "${negative}")
  if(NOT EXISTS "${input}")
    message(FATAL_ERROR "Download the companion assets before CLI tests: ${input}")
  endif()
endforeach()

function(run_cli label expected_exit)
  execute_process(
    COMMAND "${EXECUTABLE}" ${ARGN}
    WORKING_DIRECTORY "${working_dir}"
    RESULT_VARIABLE actual_exit
    OUTPUT_VARIABLE standard_output
    ERROR_VARIABLE standard_error
    TIMEOUT 120)
  file(WRITE "${run_dir}/${label}.log"
    "Exit: ${actual_exit}\nSTDOUT:\n${standard_output}\nSTDERR:\n${standard_error}")
  if(NOT "${actual_exit}" STREQUAL "${expected_exit}")
    message(FATAL_ERROR
      "${label}: expected exit ${expected_exit}, got ${actual_exit}\n"
      "${standard_output}\n${standard_error}")
  endif()
  set("${label}_STDOUT" "${standard_output}" PARENT_SCOPE)
endfunction()

# Argument and input errors must fail before loading any learned model.
run_cli(missing_file 2
  --reference "${run_dir}/does-not-exist.png" --moving "${scene}"
  --method sift --output-dir "${run_dir}/missing-file" --no-display)
run_cli(missing_pair_argument 2 --reference "${reference}" --method sift)
run_cli(unknown_option 2 --not-a-real-option value)
run_cli(invalid_method 2 --reference "${reference}" --moving "${scene}" --method unknown)
run_cli(invalid_threshold 2 --reference "${reference}" --moving "${scene}" --threshold nan)
run_cli(invalid_integer 2 --reference "${reference}" --moving "${scene}" --resize nonsense)
run_cli(invalid_device 2 --reference "${reference}" --moving "${scene}" --device cuda)
run_cli(conflicting_workflows 2 --demo-all --benchmark)

# The real package photograph must produce a usable positive estimate headlessly.
run_cli(positive 0
  --reference "${reference}" --moving "${scene}"
  --method sift --estimator magsac --device cpu
  --output-dir "${result_dir}" --no-display --validate)
if(NOT positive_STDOUT MATCHES "VALIDATION PASSED")
  message(FATAL_ERROR "The accepted CLI run did not print its validation marker.")
endif()

set(expected_files aligned.png matches.jpg overlay.jpg reference_boundary.jpg report.json valid_mask.png)
file(GLOB actual_files RELATIVE "${result_dir}" "${result_dir}/*")
list(SORT expected_files)
list(SORT actual_files)
if(NOT "${actual_files}" STREQUAL "${expected_files}")
  message(FATAL_ERROR "Unexpected accepted output set: ${actual_files}; expected ${expected_files}")
endif()
foreach(name IN LISTS expected_files)
  if(IS_DIRECTORY "${result_dir}/${name}")
    message(FATAL_ERROR "Expected a file, found a directory: ${name}")
  endif()
  file(SIZE "${result_dir}/${name}" size)
  if(size LESS 1)
    message(FATAL_ERROR "The CLI produced an empty output: ${name}")
  endif()
endforeach()
file(READ "${result_dir}/report.json" accepted_report)
if(NOT accepted_report MATCHES "\"accepted\"[ \t\r\n]*:[ \t\r\n]*true")
  message(FATAL_ERROR "The positive report does not contain accepted: true.")
endif()
if(NOT accepted_report MATCHES "\"opencv_version\"[ \t\r\n]*:[ \t\r\n]*\"5\\.1\\.0\"")
  message(FATAL_ERROR "The accepted report does not identify OpenCV 5.1.0.")
endif()

# A rejection in the same directory must not leave stale acceptance images.
set(note "Preserve this caller-owned note.\n")
set(nested_note "Preserve this nested caller-owned file.\n")
file(WRITE "${result_dir}/notes.txt" "${note}")
file(MAKE_DIRECTORY "${result_dir}/unrelated")
file(WRITE "${result_dir}/unrelated/keep" "${nested_note}")
run_cli(negative_reuse 3
  --reference "${reference}" --moving "${negative}"
  --method sift --estimator magsac --device cpu
  --output-dir "${result_dir}" --no-display --validate)
if(NOT negative_reuse_STDOUT MATCHES "VALIDATION FAILED")
  message(FATAL_ERROR "The rejected CLI run did not print its validation marker.")
endif()
foreach(name aligned.png valid_mask.png overlay.jpg reference_boundary.jpg)
  if(EXISTS "${result_dir}/${name}")
    message(FATAL_ERROR "Rejected rerun left a stale acceptance output: ${name}")
  endif()
endforeach()
foreach(name report.json matches.jpg notes.txt unrelated/keep)
  if(NOT EXISTS "${result_dir}/${name}" OR IS_DIRECTORY "${result_dir}/${name}")
    message(FATAL_ERROR "Rejected rerun lost an expected file: ${name}")
  endif()
  file(SIZE "${result_dir}/${name}" size)
  if(size LESS 1)
    message(FATAL_ERROR "Rejected rerun left an empty file: ${name}")
  endif()
endforeach()
file(READ "${result_dir}/notes.txt" actual_note)
file(READ "${result_dir}/unrelated/keep" actual_nested_note)
if(NOT "${actual_note}" STREQUAL "${note}" OR NOT "${actual_nested_note}" STREQUAL "${nested_note}")
  message(FATAL_ERROR "The rejected rerun changed a caller-owned sentinel.")
endif()
file(READ "${result_dir}/report.json" rejected_report)
if(NOT rejected_report MATCHES "\"accepted\"[ \t\r\n]*:[ \t\r\n]*false")
  message(FATAL_ERROR "The rejected report does not contain accepted: false.")
endif()
if(NOT rejected_report MATCHES "\"opencv_version\"[ \t\r\n]*:[ \t\r\n]*\"5\\.1\\.0\"")
  message(FATAL_ERROR "The rejected report does not identify OpenCV 5.1.0.")
endif()

message(STATUS "VALIDATION PASSED: native CLI errors, exact outputs, unrelated cwd, and rejection reuse")
