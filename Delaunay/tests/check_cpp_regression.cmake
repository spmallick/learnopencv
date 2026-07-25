# Run the real C++ CLI and verify its exact build-local artifact set.

foreach(required_variable DELAUNAY_EXE TEST_ROOT)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

# Remove stale artifacts so they cannot survive unnoticed between CTest runs.
file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")
set(output_dir "${TEST_ROOT}/output")

# Execute the bundled-data validation through the compiled entry point.
execute_process(
  COMMAND
    "${DELAUNAY_EXE}"
    --no-display
    --no-animation
    --validate
    --output-dir "${output_dir}"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_stdout
  ERROR_VARIABLE run_stderr
)
if(NOT run_result EQUAL 0)
  message(
    FATAL_ERROR
    "Delaunay regression failed (${run_result}): "
    "${run_stdout}${run_stderr}"
  )
endif()
if(NOT run_stdout MATCHES "DELAUNAY_VALIDATION_OK")
  message(FATAL_ERROR "Delaunay validation marker is missing: ${run_stdout}")
endif()

# The tutorial promises exactly two PNG artifacts and no stale extras.
file(
  GLOB output_entries
  LIST_DIRECTORIES true
  RELATIVE "${output_dir}"
  "${output_dir}/*"
)
list(SORT output_entries)
set(expected_entries "delaunay.png;voronoi.png")
if(NOT "${output_entries}" STREQUAL "${expected_entries}")
  message(
    FATAL_ERROR
    "Unexpected output set: ${output_entries}; expected ${expected_entries}"
  )
endif()

# The executable already decodes and checks shape/type; also require nonempty files.
foreach(filename delaunay.png voronoi.png)
  file(SIZE "${output_dir}/${filename}" output_size)
  if(output_size EQUAL 0)
    message(FATAL_ERROR "Generated output is empty: ${filename}")
  endif()
endforeach()
