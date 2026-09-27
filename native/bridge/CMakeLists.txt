cmake_minimum_required(VERSION 3.20)

project(
    leanmoe_bridge
    VERSION 0.1.0
    LANGUAGES C CXX
)

if(NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR "LLAMA_CPP_DIR must point to the pinned llama.cpp source tree")
endif()

get_filename_component(LLAMA_CPP_DIR "${LLAMA_CPP_DIR}" ABSOLUTE)

if(NOT EXISTS "${LLAMA_CPP_DIR}/include/llama.h")
    message(FATAL_ERROR "llama.h not found under ${LLAMA_CPP_DIR}/include")
endif()

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

# Phase 1B is an ABI/model-load smoke build.
# CUDA is deliberately disabled in CI because GitHub-hosted
# Windows runners do not provide our target NVIDIA GPU.
set(GGML_CUDA OFF CACHE BOOL "" FORCE)

add_subdirectory(
    "${LLAMA_CPP_DIR}"
    "${CMAKE_BINARY_DIR}/llama.cpp"
    EXCLUDE_FROM_ALL
)

add_library(leanmoe_bridge SHARED
    bridge.cpp
)

target_compile_features(
    leanmoe_bridge
    PRIVATE
        cxx_std_17
)

target_compile_definitions(
    leanmoe_bridge
    PRIVATE
        LEANMOE_BRIDGE_BUILD
)

target_include_directories(
    leanmoe_bridge
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${LLAMA_CPP_DIR}/include"
)

target_link_libraries(
    leanmoe_bridge
    PRIVATE
        llama
)

set_target_properties(
    leanmoe_bridge
    PROPERTIES
        OUTPUT_NAME "leanmoe_bridge"
)
