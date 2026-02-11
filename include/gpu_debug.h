#pragma once

#include <cstdio>

// Device-side debug printing utilities for matrices / sub-matrices.

__global__ void kernel_print_matrix(double *A, const size_t n, const size_t m)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            if (j != 0) printf(" ");
            printf("%.6f", A[i * m + j]);
        }
        printf("\n");
    }
}

__global__ void kernel_print_matrix(double *A, const size_t n, const size_t m, int node)
{
    printf("%d\n", node);
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            if (j != 0) printf(" ");
            printf("%.3f", A[i * m + j]);
        }
        printf("\n");
    }
}

__global__ void kernel_print_submatrix(double *A, const size_t n, const size_t m,
                                       const size_t pitch, int node)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        printf("%d\n", node);
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                if (j != 0) printf(" ");
                printf("%.3f", A[i * pitch + j]);
            }
            printf("\n");
        }
    }
}
