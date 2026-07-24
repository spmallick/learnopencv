# Verify that an intentionally invalid invocation fails cleanly and explains
# the problem. This is stronger than CTest's WILL_FAIL property, which by itself
# cannot distinguish a handled command error from a crash or signal.

cmake_minimum_required(VERSION 3.16)

foreach(_required_variable IN ITEMS EXECUTABLE TEST_ARGUMENTS EXPECTED_PATTERN)
  if(NOT DEFINED ${_required_variable} OR "${${_required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${_required_variable} must be provided")
  endif()
endforeach()

# CMake lists use semicolons, while a vertical bar is safe in all arguments
# registered by this project. Convert the transport representation back into
# the executable's individual argv entries.
string(REPLACE "|" ";" _arguments "${TEST_ARGUMENTS}")
execute_process(
  COMMAND "${EXECUTABLE}" ${_arguments}
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)
set(_log "${_stdout}${_stderr}")

# A numeric result demonstrates a normal process exit. Text such as "Subprocess
# aborted" would indicate a signal or crash and must never count as clean.
if(NOT "${_result}" MATCHES "^[0-9]+$")
  message(FATAL_ERROR
    "Invalid invocation did not exit cleanly (${_result}):\n${_log}")
endif()
if("${_result}" STREQUAL "0")
  message(FATAL_ERROR
    "Invalid invocation unexpectedly succeeded:\n${_log}")
endif()
if(NOT _log MATCHES "${EXPECTED_PATTERN}")
  message(FATAL_ERROR
    "Failure output omitted '${EXPECTED_PATTERN}':\n${_log}")
endif()

message(STATUS
  "Clean failure exit ${_result} included '${EXPECTED_PATTERN}'")
