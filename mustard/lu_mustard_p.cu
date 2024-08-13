#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <curand.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <mpi.h>

#include "argh.h"
#include "mustard.h"

#define MAX_TILE 4000
#define TILE_DIM 32

size_t N = 15 * 1;
size_t B = N / 5;
size_t T = N / B;
size_t local_width;
int myPE;
int myNodePE;
int nPE;
int verbose = 0;
int workspace = 256;
int smLimit = 20;
int runs = 1;
int numCopyStreams = 32;

int nodeToPrint = 0;
int peToPrint = 1;

__global__ void makeMatrixSymmetric(double *d_matrix, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t x = idx / n;
    size_t y = idx % n;

    if (x >= y || x >= n || y >= n)
    {
        return;
    }

    double average = 0.5 * (d_matrix[x * n + y] + d_matrix[y * n + x]);
    d_matrix[x * n + y] = average;
    d_matrix[y * n + x] = average;
}

__global__ void addIdenticalMatrix(double *d_matrix, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    d_matrix[idx * n + idx] += n;
}

// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRandomSymmetricPositiveDefiniteMatrixGPU(double *h_A, const size_t n)
{
    double *d_A;
    checkCudaErrors(cudaMalloc(&d_A, n * n * sizeof(double)));

    // Generate random matrix d_A
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    // curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    curandSetPseudoRandomGeneratorSeed(prng, 420);
    curandGenerateUniformDouble(prng, d_A, n * n);

    // d_A = (d_A + d_A^T) / 2
    size_t numThreads = 1024;
    size_t numBlocks = (N * N + numThreads - 1) / numThreads;
    makeMatrixSymmetric<<<numBlocks, numThreads>>>(d_A, N);

    // d_A = d_A + n * I
    numThreads = 1024;
    numBlocks = (N + numThreads - 1) / numThreads;
    addIdenticalMatrix<<<numBlocks, numThreads>>>(d_A, N);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_A, d_A, n * n * sizeof(double), cudaMemcpyDefault));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_A));
}

// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRSPDMatrixBlock(curandGenerator_t prng, double *d_A, const size_t n)
{
    curandGenerateUniformDouble(prng, d_A, n * n);

    // d_A = (d_A + d_A^T) / 2
    size_t numThreads = 1024;
    size_t numBlocks = (B * B + numThreads - 1) / numThreads;
    makeMatrixSymmetric<<<numBlocks, numThreads>>>(d_A, B);

    // d_A = d_A + n * I
    numThreads = 1024;
    numBlocks = (B + numThreads - 1) / numThreads;
    addIdenticalMatrix<<<numBlocks, numThreads>>>(d_A, B);

    checkCudaErrors(cudaDeviceSynchronize());
}

// Set upper triangle entries (excluding diagonal entries) in column-major order to zero.
// Then, transpose to row-major order.
void cleanCusolverLUDecompositionResult(double *L, double *U, const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            std::swap(L[i + j * n], L[i * n + j]);
            U[i * n + j] = L[i * n + j];
            L[i * n + j] = 0;
        }
        L[i * n + i] = 1;
    }
}

bool verifyLUDecomposition(double *A, double *L, double *U, const int n)
{
    auto newA = std::make_unique<double[]>(n * n);
    memset(newA.get(), 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                newA[i * n + j] += L[i * n + k] * U[k * n + j];
            }
        }
    }

    double error = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            error += fabs(A[i * n + j] - newA[i * n + j]);
        }
    }

    if (verbose)
    {
        printf("A:\n");
        printSquareMatrix(A, n);

        printf("\nnewA:\n");
        printSquareMatrix(newA.get(), n);

        printf("\nL:\n");
        printSquareMatrix(L, n);
        printf("\n");

        printf("\nU:\n");
        printSquareMatrix(U, n);
        printf("\n");
    }

    printf("error = %.6f}\n", error);

    return error <= 1e-6;
}

__global__ void kernel_print_matrix(double *A, const size_t n, const size_t m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (j != 0)
                printf(" ");
            // printf("A[%d]=%.3f", i * m + j, A[i * m + j]);
            printf("%.6f", A[i * m + j]);
        }
        printf("\n");
    }
}

__global__ void kernel_print_matrix(double *A, const size_t n, const size_t m, int node)
{
    printf("%d\n", node);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (j != 0)
                printf(" ");
            // printf("A[%d]=%.3f", i * m + j, A[i * m + j]);
            printf("%.3f", A[i * m + j]);
        }
        printf("\n");
    }
}

__global__ void kernel_print_submatrix(double *A, const size_t n, const size_t m, const size_t pitch, int node)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        printf("%d\n", node);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                if (j != 0)
                    printf(" ");
                // printf("A[%d]=%.3f", i * m + j, A[i * m + j]);
                printf("%.3f", A[i * pitch + j]);
            }
            printf("\n");
        }
    }
}

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

// TODO: create a separate class for kernels, it will have all the kernels and cublas related stuff
//?: maybe it can be shared for LU and Cholesky, or 2 children of a parent class
//  void GemmLU()
//  void TRSMLowerLU()
//  void TRSMUpperLU()
//  void GetrfLU()

void tiledLUPart(bool verify, bool dot)
{
    std::unique_ptr<double[]> originalMatrix;
    // Initialize data
    if (verify)
    {
        originalMatrix = std::make_unique<double[]>(N * N); // Column-major
        // generateRandomSymmetricPositiveDefiniteMatrixGPU(originalMatrix.get(), N);
        // if (myPE == 0 && verbose)
        //     printMatrix(originalMatrix.get(), N, N);
    }

    double *d_matrix, *d_buffer;

    size_t sliceCount = T / nPE;
    size_t sliceMax = T / nPE;
    int rem = T % nPE;
    int neighborPE = (myPE + 1) % nPE;
    if (rem > 0)
        sliceMax++;
    local_width = sliceMax * B;
    size_t buffer_width = nPE * B;
    if (rem > myPE)
        sliceCount++;
    // std::cout << T << " slices. PE " << myPE << " has " << sliceCount << std::endl;

    auto getMatrixBlock = [&](double *matrix, int i, int j, int width = local_width)
    {
        return matrix + i * B + j * B * width;
    };

    // Does it only work with symmetric?
    // tile_size is B*B; a column is tile_size*T; and there are T / nPE of them, if evenly divided;
    d_matrix = (double *)nvshmem_malloc(N * local_width * sizeof(double));
    d_buffer = (double *)nvshmem_malloc(N * buffer_width * sizeof(double));
    // d_buffer = (double *) nvshmem_malloc(N * N * sizeof(double));

    // Initialize libraries
    cublasHandle_t cublasHandle;
    checkCudaErrors(cublasCreate(&cublasHandle));
    // checkCudaErrors(cublasLoggerConfigure(verbose, verbose, 0, NULL));

    // Prepare constants
    double one = 1.0;
    double zero = 0.0;
    double minusOne = -1.0;
    
    // Needed for matrix generation only
    double *d_block;
    checkCudaErrors(cudaMalloc(&d_block, B * B * sizeof(double)));
    double *d_block_trans;
    checkCudaErrors(cudaMalloc(&d_block_trans, B * B * sizeof(double)));

    // Generate random matrix d_A
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    // curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    curandSetPseudoRandomGeneratorSeed(prng, 420 + myPE);

    for (int i = 0; i < sliceMax; i++)
    {
        if (verbose) 
            std::cout << "Generating slice " << i << std::endl;
            
        if (i < sliceCount) {
            int sliceCol = i * nPE + myPE;
            int lowerBlocks = T - sliceCol;

            for (int sliceRow = T - lowerBlocks; sliceRow < T; sliceRow++)
            {
                if (sliceRow == sliceCol)
                    generateRSPDMatrixBlock(prng, d_block, B);
                else
                    curandGenerateUniformDouble(prng, d_block, B * B);

                checkCudaErrors(cudaMemcpy2D(getMatrixBlock(d_matrix, i, sliceRow),
                                                sizeof(double) * local_width,
                                                d_block,
                                                sizeof(double) * B,
                                                sizeof(double) * B,
                                                B, cudaMemcpyHostToDevice));
                // TODO: take a look here!

                if (sliceRow != sliceCol)
                {
                    checkCudaErrors(cublasDgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, B, B, &one, d_block, B, &zero, d_block, B, d_block_trans, B)); 

                    kernel_nvshmem_put2D_multiBlock<<<108, 1024>>>(getMatrixBlock(d_matrix, sliceRow / nPE, sliceCol),
                                                                    local_width, d_block_trans, B, B, sliceRow % nPE);
                    nvshmem_quiet();
                }
            }
        }
        
        checkCudaErrors(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        if (verify) {         
            if (verbose) 
                std::cout << "Collecting slice(s) " << i << std::endl;
            checkCudaErrors(cudaMemcpy2D(getMatrixBlock(originalMatrix.get(), i*nPE + myPE, 0, N), 
                                        sizeof(double) * N, 
                                        getMatrixBlock(d_matrix, i, 0), 
                                        sizeof(double) * local_width,
                                        sizeof(double) * B, 
                                        N, cudaMemcpyDeviceToHost)); 

            for (int dstPE = 1; dstPE < nPE; dstPE++) {
                if (i * nPE + dstPE < T) {
                    kernel_nvshmem_get2D_slice_multiBlock<<<108, 1024>>>(getMatrixBlock(d_buffer, dstPE, 0), 
                                                                         buffer_width, 
                                                                         getMatrixBlock(d_matrix, i, 0), 
                                                                         local_width, B, N, dstPE);
                    nvshmem_quiet();
                    checkCudaErrors(cudaDeviceSynchronize());
                    checkCudaErrors(cudaMemcpy2D(getMatrixBlock(originalMatrix.get(), i*nPE + dstPE, 0, N), 
                                                sizeof(double) * N, 
                                                getMatrixBlock(d_buffer, dstPE, 0), 
                                                sizeof(double) * buffer_width,
                                                sizeof(double) * B, 
                                                N, cudaMemcpyDeviceToHost)); 
                }
            }
        }
    }

    // if (myPE == 0 && verbose)
    //     printMatrix(originalMatrix.get(), N, N);

    checkCudaErrors(cudaFree(d_block));
    checkCudaErrors(cudaFree(d_block_trans));

    // for (int pe = 0; pe < nPE; pe++)
    // {
    //     nvshmem_barrier_all();
    //     if (myPE == pe)
    //     {
    //         std::cout << "Data on PE " << myPE << std::endl;
    //         kernel_print_matrix<<<1, 1>>>(d_matrix, N, local_width);
    //         cudaDeviceSynchronize();
    //     }
    //     nvshmem_barrier_all();
    // }

    // Continue to initialize libraries
    cusolverDnHandle_t cusolverDnHandle;
    cusolverDnParams_t cusolverDnParams;
    checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
    checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
    checkCudaErrors(cublasSetSmCountTarget(cublasHandle, smLimit));

    // Prepare buffer for potrf
    int workspaceInBytesOnDevice;
    checkCudaErrors(cusolverDnDgetrf_bufferSize(
        cusolverDnHandle,
        B,
        B,
        d_matrix,
        N,
        &workspaceInBytesOnDevice));

    // void *h_workspace, *d_workspace_cusolver;
    double *d_workspace_cusolver;
    int workspaces = sliceCount * T;
    int *d_info;
    void **d_workspace_cublas = (void **)malloc(sizeof(void *) * workspaces);
    workspaceInBytesOnDevice *= 8;
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));
    int cublasWorkspaceSize = 1024 * workspace; // (B/256+1)*B*256*4;

    for (int i = 0; i < workspaces; i++)
    {
        checkCudaErrors(cudaMalloc(&d_workspace_cublas[i], cublasWorkspaceSize));
    }
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    int totalNodes = T;
    for (int k = 0; k < T; k++)
        for (int i = k + 1; i < T; i++)
            totalNodes += 2 + (T - (k + 1));

    if (verbose)
    {
        std::cout << "totalNodes=" << totalNodes << std::endl;
        std::cout << "bufferSize=" << workspaceInBytesOnDevice << std::endl;
        std::cout << "tileSize=" << cublasWorkspaceSize << std::endl;
        std::cout << "workspaces=" << workspaces << std::endl;
    }

    // setup streams and events for graph construction
    cudaStream_t s;
    checkCudaErrors(cudaStreamCreate(&s));
    numCopyStreams = 128;
    cudaStream_t copyStreams[numCopyStreams];
    cudaEvent_t copyEvents[numCopyStreams + 1];
    checkCudaErrors(cudaEventCreate(&copyEvents[numCopyStreams]));
    for (int streamID = 0; streamID < numCopyStreams; streamID++)
    {
        checkCudaErrors(cudaStreamCreate(&copyStreams[streamID]));
        checkCudaErrors(cudaEventCreate(&copyEvents[streamID]));
    }

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
    checkCudaErrors(cublasSetStream(cublasHandle, s));
    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[0], cublasWorkspaceSize));

    int *h_dependencies;
    int *d_dependencies = (int *)nvshmem_malloc(sizeof(int) * totalNodes);
    checkCudaErrors(cudaMallocHost(&h_dependencies, sizeof(int) * totalNodes));
    for (int i = 0; i < totalNodes; i++)
        h_dependencies[i] = 0;

    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));
    auto tiledLUGraphCreator = std::make_unique<mustard::TiledGraphCreator>(s, graph, false, totalNodes);

    int nodeIndex = 0;
    auto waitHook = std::make_pair(-1, -1);
    for (int k = 0; k < T; k++)
    {
        if (verbose)
            std::cout << "CUDA Graph generation progress " << float(k)/float(T)*100.0 << "%" << std::endl;
        //* A[k][k] = GETRF(A[k][k])
        //* L[k][k]*U[k][k] = A[k][k]

        int activePE = k % nPE;
        int myNodeIndex = 0;
        // int local_k = max(0, k - myPE)/nPE;
        int local_k = k / nPE;
        int distance = myPE - activePE;
        if (distance < 0)
            distance += nPE;

        if (activePE == myPE)
        {
            // checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[myNodeIndex++], cublasWorkspaceSize));
            tiledLUGraphCreator->beginCaptureOperation(
                std::make_pair(k, k),
                {std::make_pair(k, k)});
            checkCudaErrors(cusolverDnDgetrf(
                cusolverDnHandle,
                B,
                B,
                getMatrixBlock(d_matrix, local_k, k),
                local_width, // ?: N or local_width
                d_workspace_cusolver,
                NULL,
                d_info));

            // *: current design - check the next comment block
            if (k + 1 < T) // if there is a neighbor on the right
                mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, neighborPE, myPE);

            if (myPE == peToPrint && nodeIndex == nodeToPrint)
                kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_k, k),
                                                            B, B, local_width, nodeIndex);

            tiledLUGraphCreator->endCaptureOperation();
        }
        else if (distance < (T - k))
        { // if there are neighbor tiles
            //} else if (k+myPE < T && k+1 < T) { // if there are neighbor tiles  //! make sure it works
            if (waitHook.first == -1)
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(k, k),
                    {std::make_pair(k, k)});
            else
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(k, k),
                    {waitHook, std::make_pair(k, k)});
            // !: alternative design - add conditional while nodes OR use dependency waits on the remote
            // !: also, need to update local dependencies on the same GPU without communicating with others
            // *: current design - chain broadcast (works because immediate neighbor's tiles are more important)
            // *also distributes dependency update load and does not require busy-waiting on remote memory
            mustard::kernel_dep_wait<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, myPE);
            h_dependencies[nodeIndex]++;
            if ((k % nPE) != neighborPE && distance + 1 < (T - k)) // if there's a neighbor and it's not the parent
                mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, neighborPE, myPE);
            // nvshmemx_quiet_on_stream(s); // need this for consistency

            kernel_nvshmem_get2D<<<1, 1024, 0, s>>>(getMatrixBlock(d_buffer, k % nPE, k, buffer_width), buffer_width,
                                                    getMatrixBlock(d_matrix, local_k, k), local_width,
                                                    B, k % nPE);
            // checkCudaErrors(cudaEventRecord(copyEvents[numCopyStreams], s));
            // for (int row = 0; row < B; row++)
            // {
            //     int idx = row % numCopyStreams;
            //     checkCudaErrors(cudaStreamWaitEvent(copyStreams[idx], copyEvents[numCopyStreams], 0));
            //     nvshmemx_double_get_on_stream(getMatrixBlock(d_buffer, k % nPE, k, buffer_width) + buffer_width * row,
            //                                   getMatrixBlock(d_matrix, local_k, k) + local_width * row,
            //                                   B, k % nPE, copyStreams[idx]);
            //     checkCudaErrors(cudaEventRecord(copyEvents[idx], copyStreams[idx]));
            //     checkCudaErrors(cudaStreamWaitEvent(s, copyEvents[idx], 0));
            // }
            nvshmemx_quiet_on_stream(s);
            tiledLUGraphCreator->endCaptureOperation();
        }
        nodeIndex++;

        for (int i = k + 1; i < T; i++)
        {
            //* L[i][k] = TRSM(A[i][k], A[k][k]) // the U part of A[k][k]
            // seemed like only these needed a separate workspace previously, //! not sure now
            //* no need for local_i version because everything here is local
            //* as all the tiles belong to the same column and therefore same PE

            if (activePE == myPE)
            {
                // ?: there seemed to be some issue with this workspace thingy
                checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[myNodeIndex++],
                                                   cublasWorkspaceSize));
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(k, i),
                    {std::make_pair(k, k), std::make_pair(k, i)});
                checkCudaErrors(cublasDtrsm(
                    cublasHandle,
                    CUBLAS_SIDE_LEFT, // used to be right for cholesky
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,      // CUBLAS_OP_T for cholesky
                    CUBLAS_DIAG_UNIT, // CUBLAS_DIAG_NON_UNIT for cholesky
                    B, B,
                    &one,
                    getMatrixBlock(d_matrix, local_k, k), local_width,
                    getMatrixBlock(d_matrix, local_k, i), local_width));

                if (myPE == peToPrint && nodeIndex == nodeToPrint)
                    kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_k, i),
                                                                B, B, local_width, nodeIndex);

                if (k + 1 < T) // if there is a neighbor on the right
                    mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, neighborPE, myPE);
                tiledLUGraphCreator->endCaptureOperation();
            }
            else if (distance < (T - k))
            { // if there are neighbor tiles
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(k, i),
                    {std::make_pair(k, k), std::make_pair(k, i)});
                mustard::kernel_dep_wait<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, myPE);
                h_dependencies[nodeIndex]++;
                // if there's a neighbor and it's not the parent, update their dependencies too (chain)
                // if (k+myPE+1 < T && (k % nPE) != neighborPE)
                if ((k % nPE) != neighborPE && distance + 1 < (T - k)) // if there's a neighbor and it's not the parent
                    mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex,
                                                                   neighborPE, myPE);
                // nvshmemx_quiet_on_stream(s);
                kernel_nvshmem_get2D<<<1, 1024, 0, s>>>(getMatrixBlock(d_buffer, k % nPE, i, buffer_width), buffer_width,
                                                        getMatrixBlock(d_matrix, local_k, i), local_width,
                                                        B, k % nPE);
                // checkCudaErrors(cudaEventRecord(copyEvents[numCopyStreams], s));
                // for (int row = 0; row < B; row++)
                // {
                //     int idx = row % numCopyStreams;
                //     checkCudaErrors(cudaStreamWaitEvent(copyStreams[idx], copyEvents[numCopyStreams], 0));
                //     nvshmemx_double_get_on_stream(getMatrixBlock(d_buffer, k % nPE, i, buffer_width) + buffer_width * row,
                //                                   getMatrixBlock(d_matrix, local_k, i) + local_width * row,
                //                                   B, k % nPE, copyStreams[idx]);
                //     checkCudaErrors(cudaEventRecord(copyEvents[idx], copyStreams[idx]));
                //     checkCudaErrors(cudaStreamWaitEvent(s, copyEvents[idx], 0));
                // }
                nvshmemx_quiet_on_stream(s);
                tiledLUGraphCreator->endCaptureOperation();
            }
            nodeIndex++;
        }

        for (int i = k + 1; i < T; i++)
        {
            //* U[k][i] = TRSM(A[k][k], A[k][i]) // the L part of A[k][k]

            activePE = i % nPE;
            int local_i = i / nPE;

            if (activePE == myPE)
            {
                checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[myNodeIndex++],
                                                   cublasWorkspaceSize));
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(i, k),
                    {std::make_pair(k, k), std::make_pair(i, k)});

                // *: required memcpy for TRSM_RIGHT: needs the GETRF output
                if (k % nPE != myPE)
                { // this should only be copied if the parent is remote
                    checkCudaErrors(cublasDtrsm(
                        cublasHandle,
                        CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        B, B,
                        &one,
                        getMatrixBlock(d_buffer, k % nPE, k, buffer_width), buffer_width,
                        getMatrixBlock(d_matrix, local_i, k), local_width));
                }
                else
                {
                    checkCudaErrors(cublasDtrsm(
                        cublasHandle,
                        CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        B, B,
                        &one,
                        getMatrixBlock(d_matrix, local_k, k), local_width,   // k + k * N;
                        getMatrixBlock(d_matrix, local_i, k), local_width)); // (i + B) + k * N;
                }
                if (myPE == peToPrint && nodeIndex == nodeToPrint)
                {
                    kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_i, k),
                                                                B, B, local_width, nodeIndex);
                }
                tiledLUGraphCreator->endCaptureOperation();
            }
            nodeIndex++;

            for (int j = k + 1; j < T; j++)
            {
                //* A[j][i] = GEMM(A[j][k], A[i][k])
                //* A[j][i] = A[j][i] - L[j][k] * L[i][k]^T

                if (activePE == myPE)
                {
                    if (waitHook.first != i)
                    {
                        waitHook = std::make_pair(i, j);
                        // std::cout << "waitHook set to " << waitHook << std::endl;
                    }
                    checkCudaErrors(cublasSetWorkspace(cublasHandle, d_workspace_cublas[myNodeIndex++],
                                                       cublasWorkspaceSize));
                    tiledLUGraphCreator->beginCaptureOperation(
                        std::make_pair(i, j),
                        {std::make_pair(i, k), std::make_pair(k, j), std::make_pair(i, j)});
                    // *: required memcpy for GEMM: needs the TRSM_LEFT output (TRSM_RIGHT is local)
                    if (k % nPE != myPE)
                    { // this should only be copied if the parent is remote
                        checkCudaErrors(cublasGemmEx(
                            cublasHandle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            B, B, B,
                            &minusOne,
                            getMatrixBlock(d_matrix, local_i, k), CUDA_R_64F, local_width,
                            getMatrixBlock(d_buffer, k % nPE, j, buffer_width), CUDA_R_64F, buffer_width,
                            &one,
                            getMatrixBlock(d_matrix, local_i, j), CUDA_R_64F, local_width,
                            CUBLAS_COMPUTE_64F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                    else
                    {
                        checkCudaErrors(cublasGemmEx(
                            cublasHandle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            B, B, B,
                            &minusOne,
                            getMatrixBlock(d_matrix, local_i, k), CUDA_R_64F, local_width,
                            getMatrixBlock(d_matrix, local_k, j), CUDA_R_64F, local_width,
                            &one,
                            getMatrixBlock(d_matrix, local_i, j), CUDA_R_64F, local_width,
                            CUBLAS_COMPUTE_64F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                    if (myPE == peToPrint && nodeIndex == nodeToPrint)
                    {
                        kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_i, j),
                                                                    B, B, local_width, nodeIndex);
                    }
                    tiledLUGraphCreator->endCaptureOperation();
                }
                nodeIndex++;
            }
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());

    for (int pe = 0; pe < nPE; pe++)
    {
        nvshmem_barrier_all();
        if (dot && myPE == pe)
        {
            std::cout << "Printing .dot graphs on PE " << myPE << "..." << std::endl;
            char filename1[20];
            sprintf(filename1, "./graph_%d_PE%d.dot", 0, myPE);
            checkCudaErrors(cudaGraphDebugDotPrint(tiledLUGraphCreator->graph, filename1, 0));
        }
    }

    // for (int pe = 0; pe < nPE; pe++) {
    //     nvshmem_barrier_all();
    //     if (myPE == pe) {
    //         std::cout << " PE " << myPE << std::endl;
    //         for (int i = 0; i < totalNodes; i++)
    //         {
    //             if (h_dependencies[i] > 0)
    //                 std::cout << i << ":" << h_dependencies[i] << std::endl;
    //         }
    //     }
    // }

    checkCudaErrors(cudaMemcpy((void *)d_dependencies, (void *)h_dependencies,
                               sizeof(int) * totalNodes, cudaMemcpyHostToDevice));

    if (verbose)
        std::cout << "Instantiate graph..." << std::endl;
    cudaGraphExec_t graphExec;
    CudaEventClock clock;
    double totalTime = 0.0;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, tiledLUGraphCreator->graph, NULL));

    if (verbose)
        showMemUsage();
    if (verbose)
        std::cout << "Launching..." << std::endl;

    for (int i = 0; i < runs; i++)
    {

        nvshmem_barrier_all();
        clock.start(s);
        checkCudaErrors(cudaGraphLaunch(graphExec, s));
        checkCudaErrors(cudaStreamSynchronize(s));
        clock.end(s);
        checkCudaErrors(cudaDeviceSynchronize());
        printf("device %d | %d run finished\n", myPE, i);
        nvshmem_barrier_all();

        checkCudaErrors(cudaMemcpy((void *)d_dependencies, (void *)h_dependencies,
                                   sizeof(int) * totalNodes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        float time = clock.getTimeInSeconds();
        printf("device %d | %d run | time (s): %4.4f\n", myPE, i, time);
        totalTime += time;

        // for (int pe = 0; pe < nPE; pe++) {
        //     nvshmem_barrier_all();
        //     if (myPE == pe) {
        //         std::cout << " PE " << myPE << std::endl;
        //         kernel_print_matrix<<<1, 1>>>(d_matrix, N, local_width);
        //         cudaDeviceSynchronize();
        //     }
        // }
    }
    nvshmem_barrier_all();
    if (verbose)
        std::cout << "Done" << std::endl;

    nvshmem_barrier_all();
    if (verify)
    {
        double *h_L = (double *)malloc(N * N * sizeof(double));
        double *h_U = (double *)malloc(N * N * sizeof(double));
        // copy the local tiles
        for (int i = 0; i < sliceCount; i++)
        {
            checkCudaErrors(cudaMemcpy2D(getMatrixBlock(h_L, i * nPE + myPE, 0, N),
                                         sizeof(double) * N,
                                         getMatrixBlock(d_matrix, i, 0),
                                         sizeof(double) * local_width,
                                         sizeof(double) * B,
                                         N, cudaMemcpyDeviceToHost));
        }
        // std::cout << std::endl << "After local copy:" << std::endl;
        // printSquareMatrix(h_L, N);
        // copy the remote tiles
        for (int iPE = 1; iPE < nPE; iPE++)
        {

            nvshmem_double_get(d_matrix, d_matrix, local_width * N, iPE);
            int remoteSliceCount = T / nPE + (T % nPE > iPE);
            if (verbose)
                std::cout << "Collect " << remoteSliceCount << " slices from PE " << iPE << std::endl;
            for (int i = 0; i < remoteSliceCount; i++)
            {
                checkCudaErrors(cudaMemcpy2D(getMatrixBlock(h_L, i * nPE + iPE, 0, N),
                                             sizeof(double) * N,
                                             getMatrixBlock(d_matrix, i, 0),
                                             sizeof(double) * local_width,
                                             sizeof(double) * B,
                                             N, cudaMemcpyDeviceToHost));
                // std::cout << std::endl << "After slice " << i*nPE + iPE << " :" << std::endl;
                // printSquareMatrix(h_L, N);
            }

            // std::cout << "buffer of  PE " << myPE << std::endl;
            // kernel_print_matrix<<<1, 1>>>(d_buffer, N, B);
            // cudaDeviceSynchronize();
        }
        // checkCudaErrors(cudaMemcpy(h_L, d_matrix, N * N * sizeof(double), cudaMemcpyDeviceToHost));
        memset(h_U, 0, N * N * sizeof(double));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n", verifyLUDecomposition(originalMatrix.get(), h_L, h_U, N));

        free(h_L);
        free(h_U);
    }
    printf("Total time used (s): %4.4f\n", totalTime);

    nvshmem_free(d_matrix);
    nvshmem_free(d_dependencies);
    checkCudaErrors(cudaFreeHost(h_dependencies));
    checkCudaErrors(cudaFree(d_info));
    // checkCudaErrors(cudaFree(h_workspace));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++)
    {
        checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    }
}

int main(int argc, char **argv)
{
    auto cmdl = argh::parser(argc, argv);

    if (!(cmdl({"N", "n"}, N) >> N))
    {
        std::cerr << "Must provide a valid N value! Got '" << cmdl({"N", "n"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"t", "T"}, T) >> T))
    {
        std::cerr << "Must provide a valid T value! Got '" << cmdl({"T", "t"}).str() << "'" << std::endl;
        return 0;
    }
    if (N % T > 0)
    {
        std::cerr << "N must be divisible by T! Got 'N=" << N << " & T=" << T << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"sm", "SM", "smLimit"}, smLimit) >> smLimit) || smLimit > 108 || smLimit < 1)
    {
        std::cerr << "Must provide a valid SM Limit value! Got '" << cmdl({"sm", "SM", "smLimit"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"workspace", "ws", "w", "W"}, workspace) >> workspace) || workspace > 1024 * 1024 || workspace < 1)
    {
        std::cerr << "Must provide a valid workspace (in kBytes) value! Got '" << cmdl({"workspace", "ws", "w"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"run", "runs", "r", "R"}, runs) >> runs) || runs < 1)
    {
        std::cerr << "Must provide a valid number of runs! Got '" << cmdl({"run", "r", "R"}).str() << "'" << std::endl;
        return 0;
    }
    if (!(cmdl({"p", "P"}, nodeToPrint) >> nodeToPrint) || nodeToPrint < -1)
    {
        std::cerr << "Unknown" << std::endl;
        return 0;
    }

    nvshmem_init();
    myPE = nvshmem_my_pe();
    myNodePE = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    nPE = nvshmem_n_pes();
    checkCudaErrors(cudaSetDevice(myNodePE));

    int gpusAvailable = -1;
    checkCudaErrors(cudaGetDeviceCount(&gpusAvailable));
    verbose = cmdl[{"v", "verbose"}] && myPE == 0;

    // if (verbose) {
    printf("Hello from NVSHMEM_PE=%d/%d\n", myPE, nPE);
    // printf("%d GPUs detected, asked to use use %d GPUs\n", gpusAvailable, nPE);
    // }
    // if (gpusAvailable < nPE)
    //     nPE = gpusAvailable;

    B = N / T;

    if (myPE == 0)
    {
        std::cout << " with N=" << N << " (" << T << " of " << B << "x" << B << " tiles)" << std::endl;
        std::cout << "SM Limit per kernel = " << smLimit << std::endl;
        std::cout << "cuBLAS workspace = " << workspace << " kB" << std::endl;
    }

    if (nPE < 2)
    {
        std::cout << "ONLY WORKS FOR MULTI-GPU! RUN WITH MULTIPLE MPI PROCESSES" << std::endl;
        return 0;
    }

    tiledLUPart(cmdl["verify"] && myPE == 0, cmdl["dot"]);
    // testLUPart(cmdl["verify"] && myPE==0, cmdl["dot"]);

    nvshmem_finalize();

    return 0;
}

// nvcc lu_partg_nvshmem.cu -I${NVSHMEM_PATH}/include -L${NVSHMEM_PATH}/lib -lcublas -lcusolver -lnvshmem -lnvidia-ml -o nvshmem_lu
