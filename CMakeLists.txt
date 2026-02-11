cmake_minimum_required(VERSION 3.23)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(mustard VERSION 1.0
        DESCRIPTION "MUlti-gpu Scheduling of TAsk gRaphs on the Device"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# CUDA
set(MUSTARD_CUDA_MIN_VERSION "12.3" CACHE STRING "Minimum required CUDA toolkit version")
set(MUSTARD_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for (e.g. 80, 80;90, native)")

find_package(CUDAToolkit ${MUSTARD_CUDA_MIN_VERSION} REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(MUSTARD_CUDA_ARCHITECTURES STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    set(CMAKE_CUDA_ARCHITECTURES "${MUSTARD_CUDA_ARCHITECTURES}")
endif()

message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures:   ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES -DVERBOSE")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_compile_options(-Wno-deprecated-declarations)

include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/include")

# Baselines (off by default — enable with -DMUSTARD_BUILD_BASELINES=ON)
option(MUSTARD_BUILD_BASELINES "Build baseline implementations (cuSOLVER_Mg, StarPU, SLATE)" OFF)
if(MUSTARD_BUILD_BASELINES)
    # cuSOLVER multi-GPU baselines (always available with CUDA)
    if(EXISTS "${CMAKE_SOURCE_DIR}/baselines/cusolver_Mg/CMakeLists.txt")
        add_subdirectory(baselines/cusolver_Mg)
    endif()

    # StarPU baselines (requires StarPU via pkg-config; skipped if not found)
    if(EXISTS "${CMAKE_SOURCE_DIR}/baselines/starpu/lu/CMakeLists.txt")
        add_subdirectory(baselines/starpu/lu)
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/baselines/starpu/cholesky/CMakeLists.txt")
        add_subdirectory(baselines/starpu/cholesky)
    endif()

    # SLATE baselines (requires SLATE via cmake find_package; skipped if not found)
    if(EXISTS "${CMAKE_SOURCE_DIR}/baselines/slate/CMakeLists.txt")
        add_subdirectory(baselines/slate)
    endif()
endif()

add_subdirectory(mustard)