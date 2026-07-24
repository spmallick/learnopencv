if(NOT DEFINED EXECUTABLE OR NOT DEFINED INPUT OR NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR
    "EXECUTABLE, INPUT, and OUTPUT_DIR are required.")
endif()

# Hash the disposable build-directory copy before exercising the guard.
file(SHA256 "${INPUT}" input_hash_before)

# Ask the executable to use that copy as both its input and output.
execute_process(
  COMMAND "${EXECUTABLE}"
    --input "${INPUT}"
    --output-dir "${OUTPUT_DIR}"
    --output-name "collision-input.mp4"
    --no-display
  RESULT_VARIABLE result
  OUTPUT_VARIABLE standard_output
  ERROR_VARIABLE standard_error)

# The guard must reject the command with a concise, specific diagnostic.
if(result EQUAL 0)
  message(FATAL_ERROR
    "The executable accepted identical input and output paths.")
endif()
set(combined_output "${standard_output}\n${standard_error}")
string(FIND "${combined_output}"
  "input and output videos must use different paths" message_position)
if(message_position EQUAL -1)
  message(FATAL_ERROR
    "The collision error did not contain the expected diagnostic.")
endif()

# Prove that rejection happened before VideoWriter could truncate the input.
file(SHA256 "${INPUT}" input_hash_after)
if(NOT input_hash_before STREQUAL input_hash_after)
  message(FATAL_ERROR
    "The input changed while checking the collision guard.")
endif()

message(STATUS "COLLISION GUARD PASSED: input bytes unchanged")
