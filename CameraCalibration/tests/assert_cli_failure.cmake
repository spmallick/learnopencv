# Require the caller to identify the public executable under test.
if(NOT DEFINED PROGRAM)
  message(FATAL_ERROR "PROGRAM was not provided")
endif()

# Build the child command from the numbered arguments supplied by CTest.
set(command "${PROGRAM}")
foreach(argument_index RANGE 1 8)
  if(DEFINED ARGUMENT_${argument_index})
    list(APPEND command "${ARGUMENT_${argument_index}}")
  endif()
endforeach()

# Capture both output streams and the real process status without a shell.
execute_process(
  COMMAND ${command}
  RESULT_VARIABLE process_result
  OUTPUT_VARIABLE process_stdout
  ERROR_VARIABLE process_stderr
)

# A failure-path regression must never accept a successful exit.
if(process_result EQUAL 0)
  message(FATAL_ERROR "Command unexpectedly succeeded: ${command}")
endif()

# Require the intended diagnostic so an unrelated crash cannot pass the test.
set(process_output "${process_stdout}${process_stderr}")
if(NOT process_output MATCHES "${EXPECTED_ERROR}")
  message(
    FATAL_ERROR
    "Expected diagnostic '${EXPECTED_ERROR}', got:\n${process_output}"
  )
endif()

# Emit one compact marker that remains visible in verbose CTest output.
message(STATUS "Observed expected CLI failure: ${EXPECTED_ERROR}")
