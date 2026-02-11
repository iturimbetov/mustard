#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>

// NVSHMEM 2D data transfer kernels.
// These provide device-side NVSHMEM get/put operations on 2D sub-matrices.

__global__ void kernel_nvshmem_put2D(double *dst_data, size_t dst_width,
                                     double *src_data, size_t src_width,
                                     const size_t B, const int PE)
{
    for (int row = 0; row < B; row++)
    {
        nvshmemx_double_put_block(dst_data + dst_width * row,
                                  src_data + src_width * row,
                                  B, PE);
    }
}

__global__ void kernel_nvshmem_put2D_multiBlock(double *dst_data, size_t dst_width,
                                                double *src_data, size_t src_width,
                                                const size_t B, const int PE)
{
    for (int row = 0; row < B; row++)
    {
        if (row % gridDim.x == blockIdx.x)
            nvshmemx_double_put_block(dst_data + dst_width * row,
                                      src_data + src_width * row,
                                      B, PE);
    }
}

__global__ void kernel_nvshmem_get2D(double *dst_data, size_t dst_width,
                                     double *src_data, size_t src_width,
                                     const size_t B, const int PE)
{
    for (int row = 0; row < B; row++)
    {
        nvshmemx_double_get_block(dst_data + dst_width * row,
                                  src_data + src_width * row,
                                  B, PE);
    }
}

__global__ void kernel_nvshmem_get2D_multiBlock(double *dst_data, size_t dst_width,
                                                double *src_data, size_t src_width,
                                                const size_t B, const int PE)
{
    for (int row = 0; row < B; row++)
    {
        if (row % gridDim.x == blockIdx.x)
            nvshmemx_double_get_block(dst_data + dst_width * row,
                                      src_data + src_width * row,
                                      B, PE);
    }
}

__global__ void kernel_nvshmem_get2D_slice_multiBlock(double *dst_data, size_t dst_width,
                                                      double *src_data, size_t src_width,
                                                      const size_t B, const size_t N, const int PE)
{
    for (int row = 0; row < N; row++)
    {
        if (row % gridDim.x == blockIdx.x)
            nvshmemx_double_get_block(dst_data + dst_width * row,
                                      src_data + src_width * row,
                                      B, PE);
    }
}
