cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # RTX 4090 (Ada Lovelace). Set this before project() so CMake does not
  # initialize its legacy compiler default (which may be sm_52).
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(ice_carver VERSION 0.1.1 LANGUAGES CXX CUDA)

option(ICECARVER_ENABLE_TARGET "Build the private optimized target solver" OFF)
option(ICECARVER_BUILD_TESTS "Build lightweight invariant tests" ON)
set(ICECARVER_SOLUTION_SOURCE "handout/src/solution.cu" CACHE FILEPATH
    "Contestant solution translation unit")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(ICECARVER_COMMON_WARNINGS
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;-lineinfo;-Wno-deprecated-gpu-targets>)

if(NOT IS_ABSOLUTE "${ICECARVER_SOLUTION_SOURCE}")
  set(ICECARVER_SOLUTION_SOURCE
      "${CMAKE_CURRENT_SOURCE_DIR}/${ICECARVER_SOLUTION_SOURCE}")
endif()
if(NOT EXISTS "${ICECARVER_SOLUTION_SOURCE}")
  message(FATAL_ERROR "ICECARVER_SOLUTION_SOURCE does not exist: ${ICECARVER_SOLUTION_SOURCE}")
endif()

add_library(icecarver_solution OBJECT "${ICECARVER_SOLUTION_SOURCE}")
add_library(icecarver_reference OBJECT reference/cuda_reference.cu)

set(ICECARVER_EVALUATOR_SOURCES
    evaluator/main.cu
    evaluator/config_parser.cpp
    evaluator/validator.cu
    generator/field_generator.cu
    reference/cpu_reference.cpp)

add_executable(icecarver_eval
    ${ICECARVER_EVALUATOR_SOURCES}
    $<TARGET_OBJECTS:icecarver_solution>
    $<TARGET_OBJECTS:icecarver_reference>)

if(ICECARVER_ENABLE_TARGET)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/cuda_target.cu")
    message(FATAL_ERROR "ICECARVER_ENABLE_TARGET=ON but reference/cuda_target.cu is absent")
  endif()
  add_library(icecarver_target OBJECT reference/cuda_target.cu)
  target_include_directories(icecarver_target PRIVATE include)
  target_compile_options(icecarver_target PRIVATE ${ICECARVER_COMMON_WARNINGS})
  target_sources(icecarver_eval PRIVATE $<TARGET_OBJECTS:icecarver_target>)
  target_compile_definitions(icecarver_eval PRIVATE ICECARVER_ENABLE_TARGET=1)
endif()

foreach(target icecarver_solution icecarver_reference)
  target_include_directories(${target} PRIVATE include)
  target_compile_options(${target} PRIVATE ${ICECARVER_COMMON_WARNINGS})
endforeach()

target_include_directories(icecarver_eval PRIVATE include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(icecarver_eval PRIVATE ${ICECARVER_COMMON_WARNINGS})

target_link_libraries(icecarver_eval PRIVATE CUDA::cudart Threads::Threads)
set_target_properties(icecarver_eval PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

install(TARGETS icecarver_eval RUNTIME DESTINATION bin)

if(ICECARVER_BUILD_TESTS)
  enable_testing()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/table_invariants.cu")
    add_executable(icecarver_table_invariants tests/table_invariants.cu)
    target_include_directories(icecarver_table_invariants PRIVATE include)
    target_compile_options(icecarver_table_invariants PRIVATE ${ICECARVER_COMMON_WARNINGS})
    target_link_libraries(icecarver_table_invariants PRIVATE CUDA::cudart)
    add_test(NAME table_invariants COMMAND icecarver_table_invariants)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/guarded_output_contract.cu")
    add_executable(icecarver_guarded_output_contract
        tests/guarded_output_contract.cu evaluator/validator.cu)
    target_include_directories(icecarver_guarded_output_contract PRIVATE
        include ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(icecarver_guarded_output_contract PRIVATE
        ${ICECARVER_COMMON_WARNINGS})
    target_link_libraries(icecarver_guarded_output_contract PRIVATE
        CUDA::cudart)
    add_test(NAME guarded_output_contract COMMAND icecarver_guarded_output_contract)
  endif()
endif()
