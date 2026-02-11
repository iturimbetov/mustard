find_package(PkgConfig)
pkg_check_modules(STARPU starpu-1.3)
if(NOT STARPU_FOUND)
    message(WARNING "StarPU not found. StarPU LU baseline will be skipped.\n"
                    "  Set PKG_CONFIG_PATH to include your StarPU installation's pkgconfig directory.")
    return()
endif()

find_package(BLAS REQUIRED)

add_executable(starpu_lu_example
    lu_example_double.c
    dlu.c
    dlu_pivot.c
    dlu_kernels.c
    ../common/blas.c)

target_compile_definitions(starpu_lu_example PRIVATE STARPU_OPENBLAS)
target_include_directories(starpu_lu_example PRIVATE ${STARPU_INCLUDE_DIRS})
target_link_directories(starpu_lu_example PRIVATE ${STARPU_LIBRARY_DIRS})
target_link_libraries(starpu_lu_example PRIVATE
    ${STARPU_LIBRARIES}
    ${BLAS_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::nvml
    m)
