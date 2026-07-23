foreach(required_variable EXECUTABLE PROJECT_DIR TEST_ROOT)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR "${required_variable} was not provided")
  endif()
endforeach()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/unrelated-cwd")

function(run_success label output_directory)
  execute_process(
    COMMAND
      "${EXECUTABLE}"
      ${ARGN}
      --output-dir "${output_directory}"
      --no-display
      --validate
    WORKING_DIRECTORY "${TEST_ROOT}/unrelated-cwd"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "${label} failed with exit ${result}\nstdout:\n${stdout}\nstderr:\n${stderr}"
    )
  endif()

  string(FIND "${stdout}" "Alignment validation passed" validation_position)
  if(validation_position EQUAL -1)
    message(FATAL_ERROR "${label} did not print the validation marker")
  endif()
  string(FIND
    "${stdout}"
    "Feature matches: total=500, retained=78, inliers=51"
    metrics_position
  )
  if(metrics_position EQUAL -1)
    message(FATAL_ERROR
      "${label} did not report the expected feature-match metrics\n${stdout}"
    )
  endif()

  file(GLOB output_entries
    RELATIVE "${output_directory}"
    "${output_directory}/*"
  )
  list(SORT output_entries)
  set(expected_entries aligned.jpg matches.jpg)
  if(NOT "${output_entries}" STREQUAL "${expected_entries}")
    message(FATAL_ERROR
      "${label} output manifest was '${output_entries}', expected '${expected_entries}'"
    )
  endif()

  foreach(output_file IN LISTS expected_entries)
    file(SIZE "${output_directory}/${output_file}" output_size)
    if(output_size EQUAL 0)
      message(FATAL_ERROR "${label} wrote an empty ${output_file}")
    endif()
  endforeach()

  message("${stdout}")
endfunction()

function(run_failure label expected_error)
  execute_process(
    COMMAND "${EXECUTABLE}" ${ARGN}
    WORKING_DIRECTORY "${TEST_ROOT}/unrelated-cwd"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  if(NOT result EQUAL 1)
    message(FATAL_ERROR
      "${label} returned '${result}', expected a clean exit status of 1"
    )
  endif()
  string(FIND "${stderr}" "${expected_error}" error_position)
  if(error_position EQUAL -1)
    message(FATAL_ERROR
      "${label} did not report '${expected_error}'\nstdout:\n${stdout}\nstderr:\n${stderr}"
    )
  endif()
endfunction()

run_success(default_inputs "${TEST_ROOT}/default")
run_success(
  explicit_inputs
  "${TEST_ROOT}/nested/explicit"
  --input "${PROJECT_DIR}/scanned-form.jpg"
  --reference "${PROJECT_DIR}/form.jpg"
)

run_failure(
  missing_input
  "Could not read input image"
  --input "${TEST_ROOT}/missing-input.jpg"
  --output-dir "${TEST_ROOT}/failure-output"
)
run_failure(
  missing_reference
  "Could not read reference image"
  --reference "${TEST_ROOT}/missing-reference.jpg"
  --output-dir "${TEST_ROOT}/failure-output"
)
run_failure(unknown_argument "Unknown argument" --unknown)
run_failure(missing_option_value "Missing value for --input" --input)
