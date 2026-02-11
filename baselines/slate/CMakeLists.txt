# SLATE baseline: LU and Cholesky via SLATE library.
# Requires CMAKE_PREFIX_PATH or slate_DIR pointing to the SLATE installation.
# Example: -DCMAKE_PREFIX_PATH=/opt/slate
#      or: -Dslate_DIR=/opt/slate/lib/cmake/slate
#
# If SLATE was built against MKL, set MKLROOT or MKL_DIR so the linker can
# find the MKL shared libraries (e.g. -DMKLROOT=/opt/intel/oneapi/2024.1).

find_package(slate QUIET)
if(NOT slate_FOUND)
    message(WARNING "SLATE not found. SLATE baselines will be skipped.\n"
                    "  Set slate_DIR to the directory containing slateConfig.cmake\n"
                    "  (e.g. -Dslate_DIR=/opt/slate/lib/cmake/slate)")
    return()
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# SLATE/blaspp may have been built against MKL.  Make the MKL library
# directory available to the linker so -lmkl_* flags can resolve.
if(DEFINED ENV{MKLROOT})
    link_directories("$ENV{MKLROOT}/lib/intel64" "$ENV{MKLROOT}/lib")
elseif(DEFINED MKLROOT)
    link_directories("${MKLROOT}/lib/intel64" "${MKLROOT}/lib")
else()
    # Common system locations
    foreach(_mkl_dir
        /opt/intel/oneapi/latest/lib
        /opt/intel/oneapi/2024.1/lib
        /opt/intel/mkl/lib/intel64)
        if(EXISTS "${_mkl_dir}/libmkl_core.so")
            link_directories("${_mkl_dir}")
            break()
        endif()
    endforeach()
endif()

add_executable(lu_slate lu_slate.cc)
target_include_directories(lu_slate PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lu_slate PRIVATE
    slate
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver)

add_executable(chol_slate chol_slate.cc)
target_include_directories(chol_slate PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(chol_slate PRIVATE
    slate
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver)
