# This script is invoked by CTest with an exact executable and output contract.

# Refuse incomplete test configuration instead of accidentally touching a broad path.
foreach(required_variable
        IN ITEMS EXECUTABLE OUTPUT_DIR EXPECTED_OUTPUT EXPECTED_MARKER)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

# Create the narrowly scoped test directory without recursively deleting a
# caller-provided path. Remove only the one fixed artifact this example owns;
# any unexpected stale entry remains visible and fails the exact-set check.
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(REMOVE "${OUTPUT_DIR}/${EXPECTED_OUTPUT}")

# Run the real tutorial CLI from the build directory using bundled-input defaults.
execute_process(
    COMMAND
        "${EXECUTABLE}"
        --no-display
        --validate
        --output-dir "${OUTPUT_DIR}"
    RESULT_VARIABLE example_status
    OUTPUT_VARIABLE example_stdout
    ERROR_VARIABLE example_stderr)

# Surface complete process output when the executable reports failure.
if(NOT example_status EQUAL 0)
    message(FATAL_ERROR
        "Example exited ${example_status}\n"
        "stdout:\n${example_stdout}\n"
        "stderr:\n${example_stderr}")
endif()

# A zero exit alone is insufficient; require the explicit validation marker.
string(FIND "${example_stdout}" "${EXPECTED_MARKER}" marker_position)
if(marker_position EQUAL -1)
    message(FATAL_ERROR
        "Missing validation marker '${EXPECTED_MARKER}'\n"
        "stdout:\n${example_stdout}\n"
        "stderr:\n${example_stderr}")
endif()

# Enumerate every immediate output entry, including unexpected extra artifacts.
file(GLOB output_entries
    RELATIVE "${OUTPUT_DIR}"
    LIST_DIRECTORIES true
    "${OUTPUT_DIR}/*")

# The example contract permits exactly one generated image.
list(LENGTH output_entries output_count)
if(NOT output_count EQUAL 1)
    message(FATAL_ERROR
        "Expected one output '${EXPECTED_OUTPUT}', found: ${output_entries}")
endif()

# Check the exact deterministic filename rather than merely accepting any image.
list(GET output_entries 0 actual_output)
if(NOT actual_output STREQUAL EXPECTED_OUTPUT)
    message(FATAL_ERROR
        "Expected output '${EXPECTED_OUTPUT}', found '${actual_output}'")
endif()

# The executable already decoded this file in --validate mode; also reject an
# empty artifact at the outer test layer.
file(SIZE "${OUTPUT_DIR}/${EXPECTED_OUTPUT}" output_size)
if(output_size EQUAL 0)
    message(FATAL_ERROR "Generated output is empty: ${EXPECTED_OUTPUT}")
endif()

# Preserve useful runtime and metric evidence in verbose CTest output.
message(STATUS "${example_stdout}")
