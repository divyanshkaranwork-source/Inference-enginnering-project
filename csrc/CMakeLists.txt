cmake_minimum_required(VERSION 3.16)
project(inferlab_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(INFERLAB_NATIVE "Optimise for this CPU (-march=native)" ON)
option(INFERLAB_SANITIZE "Build with AddressSanitizer + UBSan (tests)" OFF)
option(INFERLAB_OPENMP "Use OpenMP threads" ON)

add_library(inferlab_kernels SHARED
  src/matmul.cpp
  src/quant.cpp
  src/elementwise.cpp
  src/attention.cpp)
target_include_directories(inferlab_kernels PUBLIC include)
set_target_properties(inferlab_kernels PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

# ---- warnings + hardening (see SECURITY.md) ----------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(inferlab_kernels PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
    -Wformat=2 -Werror=format-security
    -fstack-protector-strong -fno-strict-aliasing)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(inferlab_kernels PRIVATE -O3 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2)
  endif()
  if(INFERLAB_NATIVE AND NOT INFERLAB_SANITIZE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
      target_compile_options(inferlab_kernels PRIVATE -march=native)
    endif()
  endif()
  if(NOT APPLE)
    target_link_options(inferlab_kernels PRIVATE -Wl,-z,relro,-z,now -Wl,-z,noexecstack)
  endif()
  if(INFERLAB_SANITIZE)
    target_compile_options(inferlab_kernels PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1)
    target_link_options(inferlab_kernels PUBLIC -fsanitize=address,undefined)
  endif()
endif()

if(INFERLAB_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(inferlab_kernels PUBLIC OpenMP::OpenMP_CXX)
  else()
    message(WARNING "OpenMP not found: kernels will run single-threaded")
    target_compile_options(inferlab_kernels PRIVATE -Wno-unknown-pragmas)
  endif()
else()
  target_compile_options(inferlab_kernels PRIVATE -Wno-unknown-pragmas)
endif()

# Python loads the library from inferlab/kernels/lib/ (sanitizer builds stay in build dir)
if(NOT INFERLAB_SANITIZE)
  set(INFERLAB_PY_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../inferlab/kernels/lib")
  add_custom_command(TARGET inferlab_kernels POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${INFERLAB_PY_LIB_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:inferlab_kernels>" "${INFERLAB_PY_LIB_DIR}/"
    COMMENT "Copying kernel library into the Python package")
endif()

# ---- tests and benchmarks ------------------------------------------------------
enable_testing()
add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE inferlab_kernels)
add_test(NAME test_kernels COMMAND test_kernels)

add_executable(bench_kernels bench/bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE inferlab_kernels)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(bench_kernels PRIVATE -O2)
endif()
