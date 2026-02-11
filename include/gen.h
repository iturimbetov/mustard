#pragma once

#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

#ifdef __cplusplus
#include "utils.h"
#else
// Minimal CUDA error check for C callers that only use CPU helpers.
#include <stdio.h>
#ifndef checkCudaErrors
#define checkCudaErrors(val) \
    do { cudaError_t _e = (val); if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while(0)
#endif
#endif

// ---- CPU-side matrix generation ----

// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
static inline void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const size_t n)
{
    // srand(time(NULL));
    srand(420);

    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            h_A[i * n + j] = (double)rand() / (double)RAND_MAX;

    for (int i = 0; i < n; i++)
        for (int j = i; j >= 0; j--)
            h_A[i * n + j] = h_A[j * n + i];

    for (int i = 0; i < n; i++)
        h_A[i * n + i] = h_A[i * n + i] + n;
}

// ---- GPU-side matrix generation (CUDA only) ----

#ifdef __CUDACC__

__global__ void kernel_makeMatrixSymmetric(double *d_matrix, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t x = idx / n;
    size_t y = idx % n;

    if (x >= y || x >= n || y >= n)
        return;

    double average = 0.5 * (d_matrix[x * n + y] + d_matrix[y * n + x]);
    d_matrix[x * n + y] = average;
    d_matrix[y * n + x] = average;
}

__global__ void kernel_addIdentityScaled(double *d_matrix, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    d_matrix[idx * n + idx] += n;
}

// Generate a random SPD matrix on the GPU and copy to host.
inline void generateRandomSPDMatrixGPU(double *h_A, const size_t n)
{
    double *d_A;
    checkCudaErrors(cudaMalloc(&d_A, n * n * sizeof(double)));

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, 420);
    curandGenerateUniformDouble(prng, d_A, n * n);

    size_t numThreads = 1024;
    size_t numBlocks = (n * n + numThreads - 1) / numThreads;
    kernel_makeMatrixSymmetric<<<numBlocks, numThreads>>>(d_A, n);

    numBlocks = (n + numThreads - 1) / numThreads;
    kernel_addIdentityScaled<<<numBlocks, numThreads>>>(d_A, n);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_A, d_A, n * n * sizeof(double), cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_A));
    curandDestroyGenerator(prng);
}

// Generate a random SPD block in-place on device memory (no host copy).
inline void generateRSPDMatrixBlockGPU(curandGenerator_t prng, double *d_A, const size_t n)
{
    curandGenerateUniformDouble(prng, d_A, n * n);

    size_t numThreads = 1024;
    size_t numBlocks = (n * n + numThreads - 1) / numThreads;
    kernel_makeMatrixSymmetric<<<numBlocks, numThreads>>>(d_A, n);

    numBlocks = (n + numThreads - 1) / numThreads;
    kernel_addIdentityScaled<<<numBlocks, numThreads>>>(d_A, n);

    checkCudaErrors(cudaDeviceSynchronize());
}

#endif // __CUDACC__
