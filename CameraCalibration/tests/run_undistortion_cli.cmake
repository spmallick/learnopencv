# This CTest wrapper enforces a fresh, exact two-image output contract.

# Refuse incomplete input before touching a directory or starting a process.
foreach(
  required_variable
  IN ITEMS PROGRAM OUTPUT_ROOT OUTPUT_DIR EXPECTED_MARKER
)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "Missing required variable: ${required_variable}")
  endif()
endforeach()

# Prove that the removable target is a child of the dedicated test-output root.
get_filename_component(output_root_absolute "${OUTPUT_ROOT}" ABSOLUTE)
get_filename_component(output_directory_absolute "${OUTPUT_DIR}" ABSOLUTE)
file(
  RELATIVE_PATH output_relative
  "${output_root_absolute}"
  "${output_directory_absolute}"
)
if(output_relative STREQUAL "" OR output_relative MATCHES "^\\.\\.")
  message(
    FATAL_ERROR
    "OUTPUT_DIR must be a child of OUTPUT_ROOT: ${OUTPUT_DIR}"
  )
endif()

# CTest supplies a unique build-local directory, so clearing it rejects stale
# files without touching source assets or another test's output.
file(REMOVE_RECURSE "${output_directory_absolute}")
file(MAKE_DIRECTORY "${output_directory_absolute}")

# Run the real public executable with bundled defaults from the build directory.
execute_process(
  COMMAND
    "${PROGRAM}"
    --no-display
    --validate
    --output-dir "${output_directory_absolute}"
  RESULT_VARIABLE process_result
  OUTPUT_VARIABLE process_stdout
  ERROR_VARIABLE process_stderr
)

# Surface the child process's complete diagnostics on failure.
if(NOT process_result EQUAL 0)
  message(
    FATAL_ERROR
    "Undistortion example exited ${process_result}\n"
    "stdout:\n${process_stdout}\n"
    "stderr:\n${process_stderr}"
  )
endif()

# A zero exit is insufficient without the marker printed after all validations.
string(FIND "${process_stdout}" "${EXPECTED_MARKER}" marker_position)
if(marker_position EQUAL -1)
  message(
    FATAL_ERROR
    "Missing validation marker '${EXPECTED_MARKER}'\n"
    "stdout:\n${process_stdout}\n"
    "stderr:\n${process_stderr}"
  )
endif()

# Enumerate every immediate file or directory so an unexpected artifact fails.
file(
  GLOB output_entries
  RELATIVE "${output_directory_absolute}"
  LIST_DIRECTORIES true
  "${output_directory_absolute}/*"
)
list(SORT output_entries)

# Both methods have one fixed, documented JPEG filename.
set(
  expected_outputs
  "undistorted_direct.jpg"
  "undistorted_remap.jpg"
)
if(NOT "${output_entries}" STREQUAL "${expected_outputs}")
  message(
    FATAL_ERROR
    "Expected outputs '${expected_outputs}', found '${output_entries}'"
  )
endif()

# Reject zero-byte artifacts at the outer test layer. The executable's
# --validate path has already decoded both JPEGs and checked their dimensions.
foreach(expected_output IN LISTS expected_outputs)
  file(
    SIZE
    "${output_directory_absolute}/${expected_output}"
    output_size
  )
  if(output_size EQUAL 0)
    message(FATAL_ERROR "Generated output is empty: ${expected_output}")
  endif()
endforeach()

# Keep version, metrics, and the validation marker visible in verbose CTest.
message(STATUS "${process_stdout}")
