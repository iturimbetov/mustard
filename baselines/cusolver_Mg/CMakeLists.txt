# cuSOLVER multi-GPU baselines (LU via MgGetrf, Cholesky via MgPotrf).
# cusolverMg is found by searching the CUDA toolkit library paths.

add_executable(cusolver_MgGetrf_example cusolver_MgGetrf_example.cu)
add_executable(cusolver_MgPotrf_example cusolver_MgPotrf_example.cu)

# Locate cusolverMg library (not provided as an imported target by CUDAToolkit).
find_library(CUDA_cusolverMg_LIBRARY cusolverMg
    PATHS "${CUDAToolkit_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
if(NOT CUDA_cusolverMg_LIBRARY)
    message(FATAL_ERROR "cusolverMg library not found in ${CUDAToolkit_LIBRARY_DIR}")
endif()

target_link_libraries(cusolver_MgGetrf_example PRIVATE ${CUDA_cusolverMg_LIBRARY})
target_link_libraries(cusolver_MgPotrf_example PRIVATE ${CUDA_cusolverMg_LIBRARY})
