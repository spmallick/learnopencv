# Exercise C++ failure paths that must not mutate user-controlled inputs.

foreach(required_variable DELAUNAY_EXE SOURCE_DIR TEST_ROOT)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

# Start from a clean build-local directory on every CTest run.
file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")

# Run one CLI case that must fail and verify its diagnostic.
function(expect_cli_failure case_name expected_pattern)
  execute_process(
    COMMAND "${DELAUNAY_EXE}" ${ARGN}
    RESULT_VARIABLE case_result
    OUTPUT_VARIABLE case_stdout
    ERROR_VARIABLE case_stderr
  )
  if(case_result EQUAL 0)
    message(FATAL_ERROR "${case_name} unexpectedly succeeded")
  endif()
  if(NOT case_stderr MATCHES "${expected_pattern}")
    message(
      FATAL_ERROR
      "${case_name} did not report '${expected_pattern}': ${case_stderr}"
    )
  endif()
endfunction()

# Put the input image exactly where delaunay.png would normally be written.
set(collision_dir "${TEST_ROOT}/collision")
file(MAKE_DIRECTORY "${collision_dir}")
file(COPY "${SOURCE_DIR}/obama.jpg" DESTINATION "${collision_dir}")
file(RENAME
  "${collision_dir}/obama.jpg"
  "${collision_dir}/delaunay.png"
)
file(COPY "${SOURCE_DIR}/obama.txt" DESTINATION "${collision_dir}")

# Hash the source so the test proves that a rejected run did not overwrite it.
file(SHA256 "${collision_dir}/delaunay.png" collision_hash_before)
execute_process(
  COMMAND
    "${DELAUNAY_EXE}"
    --image "${collision_dir}/delaunay.png"
    --points "${collision_dir}/obama.txt"
    --output-dir "${collision_dir}"
    --no-display
  RESULT_VARIABLE collision_result
  OUTPUT_VARIABLE collision_stdout
  ERROR_VARIABLE collision_stderr
)
if(collision_result EQUAL 0)
  message(FATAL_ERROR "Input/output collision unexpectedly succeeded")
endif()
if(NOT collision_stderr MATCHES "would overwrite an input file")
  message(
    FATAL_ERROR
    "Collision failure did not report the safety guard: ${collision_stderr}"
  )
endif()
file(SHA256 "${collision_dir}/delaunay.png" collision_hash_after)
if(NOT collision_hash_before STREQUAL collision_hash_after)
  message(FATAL_ERROR "The colliding input image was modified")
endif()

# Different pathnames can still reference one inode through a hard link.
set(hardlink_dir "${TEST_ROOT}/hardlink")
file(MAKE_DIRECTORY "${hardlink_dir}")
file(COPY "${SOURCE_DIR}/obama.jpg" DESTINATION "${hardlink_dir}")
file(RENAME
  "${hardlink_dir}/obama.jpg"
  "${hardlink_dir}/source.jpg"
)
file(
  CREATE_LINK
  "${hardlink_dir}/source.jpg"
  "${hardlink_dir}/delaunay.png"
  RESULT hardlink_result
)
if(NOT hardlink_result STREQUAL "0")
  message(FATAL_ERROR "Could not create hard-link collision: ${hardlink_result}")
endif()
file(SHA256 "${hardlink_dir}/source.jpg" hardlink_hash_before)
execute_process(
  COMMAND
    "${DELAUNAY_EXE}"
    --image "${hardlink_dir}/source.jpg"
    --points "${SOURCE_DIR}/obama.txt"
    --output-dir "${hardlink_dir}"
    --no-display
  RESULT_VARIABLE hardlink_run_result
  OUTPUT_VARIABLE hardlink_stdout
  ERROR_VARIABLE hardlink_stderr
)
if(hardlink_run_result EQUAL 0)
  message(FATAL_ERROR "Hard-link input/output collision unexpectedly succeeded")
endif()
if(NOT hardlink_stderr MATCHES "would overwrite an input file")
  message(
    FATAL_ERROR
    "Hard-link failure did not report the safety guard: ${hardlink_stderr}"
  )
endif()
file(SHA256 "${hardlink_dir}/source.jpg" hardlink_hash_after)
if(NOT hardlink_hash_before STREQUAL hardlink_hash_after)
  message(FATAL_ERROR "The hard-linked input image was modified")
endif()

# Fractional sites are rejected because this example models integer image pixels.
set(fractional_dir "${TEST_ROOT}/fractional")
file(MAKE_DIRECTORY "${fractional_dir}")
file(
  WRITE "${fractional_dir}/points.txt"
  "10 20\n20 30\n40.5 50\n"
)
execute_process(
  COMMAND
    "${DELAUNAY_EXE}"
    --image "${SOURCE_DIR}/obama.jpg"
    --points "${fractional_dir}/points.txt"
    --output-dir "${fractional_dir}/output"
    --no-display
  RESULT_VARIABLE fractional_result
  OUTPUT_VARIABLE fractional_stdout
  ERROR_VARIABLE fractional_stderr
)
if(fractional_result EQUAL 0)
  message(FATAL_ERROR "Fractional landmark input unexpectedly succeeded")
endif()
if(NOT fractional_stderr MATCHES "coordinates must be integers")
  message(
    FATAL_ERROR
    "Fractional failure did not report the integer contract: "
    "${fractional_stderr}"
  )
endif()

# A nonexistent image must fail before any geometry or output work begins.
expect_cli_failure(
  "Missing image input"
  "Could not read input image"
  --image "${TEST_ROOT}/missing-image.jpg"
  --points "${SOURCE_DIR}/obama.txt"
  --output-dir "${TEST_ROOT}/missing-output"
  --no-display
)

# Each non-comment landmark record must contain exactly one x/y pair.
set(malformed_dir "${TEST_ROOT}/malformed")
file(MAKE_DIRECTORY "${malformed_dir}")
file(WRITE "${malformed_dir}/points.txt" "10 20 30\n")
expect_cli_failure(
  "Malformed landmark input"
  "expected two coordinates"
  --image "${SOURCE_DIR}/obama.jpg"
  --points "${malformed_dir}/points.txt"
  --output-dir "${malformed_dir}/output"
  --no-display
)

# Pixel coordinates use half-open image bounds, so x == width is invalid.
set(outside_dir "${TEST_ROOT}/outside")
file(MAKE_DIRECTORY "${outside_dir}")
file(WRITE "${outside_dir}/points.txt" "10 20\n20 30\n512 40\n")
expect_cli_failure(
  "Out-of-bounds landmark input"
  "point is outside the image rectangle"
  --image "${SOURCE_DIR}/obama.jpg"
  --points "${outside_dir}/points.txt"
  --output-dir "${outside_dir}/output"
  --no-display
)
