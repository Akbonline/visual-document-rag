cmake_minimum_required(VERSION 3.20)

project(vidore_rag_beta LANGUAGES NONE)

include(CTest)

find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)

set(
  VIDORE_TEST_MANIFEST
  "${CMAKE_SOURCE_DIR}/testlist.list"
  CACHE FILEPATH
  "Manifest containing one pytest file per line"
)

if(NOT EXISTS "${VIDORE_TEST_MANIFEST}")
  message(FATAL_ERROR "Test manifest does not exist: ${VIDORE_TEST_MANIFEST}")
endif()

file(STRINGS "${VIDORE_TEST_MANIFEST}" VIDORE_TEST_ENTRIES)

set(VIDORE_REGISTERED_TESTS 0)
set(VIDORE_DECLARED_TESTS)
foreach(TEST_ENTRY IN LISTS VIDORE_TEST_ENTRIES)
  string(STRIP "${TEST_ENTRY}" TEST_ENTRY)

  if(TEST_ENTRY STREQUAL "" OR TEST_ENTRY MATCHES "^#")
    continue()
  endif()

  if(IS_ABSOLUTE "${TEST_ENTRY}")
    message(FATAL_ERROR "Test manifest entries must be repository relative: ${TEST_ENTRY}")
  endif()

  set(TEST_PATH "${CMAKE_SOURCE_DIR}/${TEST_ENTRY}")
  if(NOT EXISTS "${TEST_PATH}")
    message(FATAL_ERROR "Test listed in manifest does not exist: ${TEST_ENTRY}")
  endif()

  list(APPEND VIDORE_DECLARED_TESTS "${TEST_ENTRY}")

  string(REGEX REPLACE "[^A-Za-z0-9_]+" "." TEST_NAME "${TEST_ENTRY}")
  string(REGEX REPLACE "^tests\." "" TEST_NAME "${TEST_NAME}")
  string(REGEX REPLACE "\.py$" "" TEST_NAME "${TEST_NAME}")

  add_test(
    NAME "pytest.${TEST_NAME}"
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${CMAKE_SOURCE_DIR}/src"
      "${Python3_EXECUTABLE}" -m pytest "${TEST_PATH}"
  )
  set_tests_properties(
    "pytest.${TEST_NAME}"
    PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  )

  math(EXPR VIDORE_REGISTERED_TESTS "${VIDORE_REGISTERED_TESTS} + 1")
endforeach()

if(VIDORE_REGISTERED_TESTS EQUAL 0)
  message(FATAL_ERROR "The test manifest did not register any tests")
endif()

file(
  GLOB VIDORE_DISCOVERED_TESTS
  CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_SOURCE_DIR}"
  "${CMAKE_SOURCE_DIR}/tests/test_*.py"
)
list(SORT VIDORE_DECLARED_TESTS)
list(SORT VIDORE_DISCOVERED_TESTS)

if(NOT VIDORE_DECLARED_TESTS STREQUAL VIDORE_DISCOVERED_TESTS)
  message(
    FATAL_ERROR
    "testlist.list must contain every tests/test_*.py file exactly once. "
    "Declared: ${VIDORE_DECLARED_TESTS}. Discovered: ${VIDORE_DISCOVERED_TESTS}."
  )
endif()

add_custom_target(
  lint
  COMMAND "${Python3_EXECUTABLE}" -m ruff check "${CMAKE_SOURCE_DIR}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Running Ruff"
  VERBATIM
)

add_custom_target(
  typecheck
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${CMAKE_SOURCE_DIR}/src"
    "${Python3_EXECUTABLE}" -m mypy "${CMAKE_SOURCE_DIR}/src"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Running mypy"
  VERBATIM
)

add_custom_target(
  check
  COMMAND
    "${CMAKE_CTEST_COMMAND}" --test-dir "${CMAKE_BINARY_DIR}" --output-on-failure
  COMMENT "Running every test registered from testlist.list"
  VERBATIM
)

message(STATUS "Registered ${VIDORE_REGISTERED_TESTS} pytest files from testlist.list")
