cmake_minimum_required(VERSION 3.16)
project(tars_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ═══════════════════════════════════════
# Platform / compiler detection
# ═══════════════════════════════════════
if(MSVC)
    # MSVC: AVX2 + fast math
    add_compile_options(/O2 /arch:AVX2 /fp:fast /W3 /EHsc)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    # GCC / Clang
    add_compile_options(-O3 -mavx2 -mfma -Wall -Wextra -ffast-math)
endif()

# ═══════════════════════════════════════
# AVX2 detection define
# ═══════════════════════════════════════
include(CheckCXXCompilerFlag)
if(MSVC)
    set(HAS_AVX2 TRUE)
else()
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
endif()
if(HAS_AVX2)
    add_compile_definitions(TARS_HAS_AVX2=1)
endif()

# ═══════════════════════════════════════
# Source files
# ═══════════════════════════════════════
set(KERNEL_SOURCES
    kernels/rmsnorm_fused.cpp
    kernels/embedding_lookup.cpp
    kernels/bitnet_matmul.cpp
    kernels/softmax_fused.cpp
    kernels/ssd_scan.cpp
    kernels/wkv7_update.cpp
    kernels/swiglu_fused.cpp
    kernels/rope_fused.cpp
    kernels/attention_diff.cpp
    kernels/smoothquant.cpp
)

set(RUNTIME_SOURCES
    runtime/arena.cpp
)

# ═══════════════════════════════════════
# tars_core shared library
# ═══════════════════════════════════════
add_library(tars_core SHARED
    ${KERNEL_SOURCES}
    ${RUNTIME_SOURCES}
)
target_include_directories(tars_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Export symbols on Windows
if(WIN32)
    target_compile_definitions(tars_core PRIVATE TARS_BUILDING_DLL)
endif()

# ═══════════════════════════════════════
# Unit tests
# ═══════════════════════════════════════
add_executable(test_kernels
    tests/test_kernels.cpp
    ${KERNEL_SOURCES}
    ${RUNTIME_SOURCES}
)
target_include_directories(test_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

enable_testing()
add_test(NAME tars_kernel_tests COMMAND test_kernels)

# ═══════════════════════════════════════
# pybind11 (optional — if found)
# ═══════════════════════════════════════
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(tars_core_py
        bindings/bind.cpp
        ${KERNEL_SOURCES}
        ${RUNTIME_SOURCES}
    )
    target_include_directories(tars_core_py PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(tars_core_py PROPERTIES
        OUTPUT_NAME "tars_core"
    )
    message(STATUS "pybind11 found — building Python bindings")
else()
    message(STATUS "pybind11 NOT found — skipping Python bindings (pip install pybind11)")
endif()
