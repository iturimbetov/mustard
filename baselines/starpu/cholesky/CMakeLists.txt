find_package(PkgConfig)
pkg_check_modules(STARPU starpu-1.3)
if(NOT STARPU_FOUND)
    message(WARNING "StarPU not found. StarPU Cholesky baseline will be skipped.\n"
                    "  Set PKG_CONFIG_PATH to include your StarPU installation's pkgconfig directory.")
    return()
endif()

find_package(BLAS REQUIRED)

add_executable(starpu_cholesky_tile_tag
    cholesky_tile_tag.c
    cholesky_models.c
    cholesky_kernels.c
    ../common/blas.c)

target_compile_definitions(starpu_cholesky_tile_tag PRIVATE STARPU_OPENBLAS)
target_include_directories(starpu_cholesky_tile_tag PRIVATE
    ${STARPU_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include)
target_link_directories(starpu_cholesky_tile_tag PRIVATE ${STARPU_LIBRARY_DIRS})
target_link_libraries(starpu_cholesky_tile_tag PRIVATE
    ${STARPU_LIBRARIES}
    ${BLAS_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::nvml
    m)
