cmake_minimum_required(VERSION 3.16)
project(chat_htm VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Forward GUI option to htm_flow
option(HTM_FLOW_WITH_GUI "Build with Qt6 GUI debugger support" OFF)

# Option to build chat_htm tests
option(CHAT_HTM_BUILD_TESTS "Build chat_htm test suite" ON)

# -----------------------------------------------------------------------------
# htm_flow submodule (provides htm_flow_core library target)
# -----------------------------------------------------------------------------
# Disable htm_flow's own test build to avoid target name collisions.
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(htm_flow)

# -----------------------------------------------------------------------------
# chat_htm executable
# -----------------------------------------------------------------------------
add_executable(chat_htm
  src/main.cpp
  src/runtime/text_runtime.cpp
)

# Our own headers live under src/ (included as "encoders/...", "text/...", etc.)
target_include_directories(chat_htm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Link against htm_flow_core (brings in all HTM algorithm + config + yaml-cpp)
target_link_libraries(chat_htm PRIVATE htm_flow_core)

# GUI support (optional)
if(HTM_FLOW_WITH_GUI)
  if(TARGET htm_gui)
    target_link_libraries(chat_htm PRIVATE htm_gui)
    target_compile_definitions(chat_htm PRIVATE HTM_FLOW_WITH_GUI)
  endif()
endif()

target_compile_features(chat_htm PRIVATE cxx_std_17)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(CHAT_HTM_BUILD_TESTS)
  # Re-use GoogleTest from htm_flow's vendored copy
  if(NOT TARGET gtest)
    add_subdirectory(htm_flow/lib/googletest)
  endif()

  enable_testing()
  include(GoogleTest)

  file(GLOB_RECURSE CHAT_HTM_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

  add_executable(chat_htm_tests ${CHAT_HTM_TEST_FILES}
    src/runtime/text_runtime.cpp
  )

  target_include_directories(chat_htm_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(chat_htm_tests PRIVATE htm_flow_core gtest gtest_main)

  # Make test data accessible: set a compile definition with the path
  target_compile_definitions(chat_htm_tests PRIVATE
    CHAT_HTM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data"
  )

  gtest_discover_tests(chat_htm_tests DISCOVERY_TIMEOUT 30)
endif()
