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
#include "cli.h"
#include "gen.h"
#include "gpu_debug.h"
#include "mustard.h"
#include "nvshmem_kernels.h"
#include "verify.h"

// Global configuration (populated from CLI in main).
static MustardConfig cfg;
static size_t& N = cfg.N;
static size_t& B = cfg.B;
static size_t& T = cfg.T;
int myPE;
static int& verbose = cfg.verbose;
static int& workspace = cfg.workspace;
static int& smLimit = cfg.smLimit;
static int& runs = cfg.runs;

// Extra globals specific to partitioned LU.
static size_t local_width;
static int myNodePE;
static int nPE;
static int numCopyStreams = 32;
static int nodeToPrint = 0;
static int peToPrint = 1;

void tiledLUPart(bool verify, bool dot)
{
    std::unique_ptr<double[]> originalMatrix;
    if (verify)
    {
        originalMatrix = std::make_unique<double[]>(N * N);
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

    auto getMatrixBlock = [&](double *matrix, int i, int j, int width = local_width)
    {
        return matrix + i * B + j * B * width;
    };

    // Tile size is B*B; a column is tile_size*T; T/nPE columns per PE if evenly divided.
    d_matrix = (double *)nvshmem_malloc(N * local_width * sizeof(double));
    d_buffer = (double *)nvshmem_malloc(N * buffer_width * sizeof(double));

    // Initialize libraries
    cublasHandle_t cublasHandle;
    checkCudaErrors(cublasCreate(&cublasHandle));

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
                    generateRSPDMatrixBlockGPU(prng, d_block, B);
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

    // Initialize libraries
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

    double *d_workspace_cusolver;
    int workspaces = sliceCount * T;
    int *d_info;
    void **d_workspace_cublas = (void **)malloc(sizeof(void *) * workspaces);
    workspaceInBytesOnDevice *= 8;
    checkCudaErrors(cudaMalloc(&d_workspace_cusolver, workspaceInBytesOnDevice));
    int cublasWorkspaceSize = 1024 * workspace;

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
        int local_k = k / nPE;
        int distance = myPE - activePE;
        if (distance < 0)
            distance += nPE;

        if (activePE == myPE)
        {
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

            // Chain broadcast: current design.
            if (k + 1 < T)
                mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, neighborPE, myPE);

            if (myPE == peToPrint && nodeIndex == nodeToPrint)
                kernel_print_submatrix<<<108, 1024, 0, s>>>(getMatrixBlock(d_matrix, local_k, k),
                                                            B, B, local_width, nodeIndex);

            tiledLUGraphCreator->endCaptureOperation();
        }
        else if (distance < (T - k))
        {
            if (waitHook.first == -1)
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(k, k),
                    {std::make_pair(k, k)});
            else
                tiledLUGraphCreator->beginCaptureOperation(
                    std::make_pair(k, k),
                    {waitHook, std::make_pair(k, k)});
            // Chain broadcast: wait for dependency, forward to neighbor if needed, then get remote data.
            mustard::kernel_dep_wait<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, myPE);
            h_dependencies[nodeIndex]++;
            if ((k % nPE) != neighborPE && distance + 1 < (T - k))
                mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex, neighborPE, myPE);

            kernel_nvshmem_get2D<<<1, 1024, 0, s>>>(getMatrixBlock(d_buffer, k % nPE, k, buffer_width), buffer_width,
                                                    getMatrixBlock(d_matrix, local_k, k), local_width,
                                                    B, k % nPE);
            nvshmemx_quiet_on_stream(s);
            tiledLUGraphCreator->endCaptureOperation();
        }
        nodeIndex++;

        for (int i = k + 1; i < T; i++)
        {
            // L[i][k] = TRSM(A[i][k], A[k][k])
            // All tiles in the same column belong to the same PE — no local_i needed.

            if (activePE == myPE)
            {
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
                if ((k % nPE) != neighborPE && distance + 1 < (T - k))
                    mustard::kernel_dep_update_noq<<<1, 1, 0, s>>>(d_dependencies, nodeIndex,
                                                                   neighborPE, myPE);
                kernel_nvshmem_get2D<<<1, 1024, 0, s>>>(getMatrixBlock(d_buffer, k % nPE, i, buffer_width), buffer_width,
                                                        getMatrixBlock(d_matrix, local_k, i), local_width,
                                                        B, k % nPE);
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
            }

        }
        memset(h_U, 0, N * N * sizeof(double));
        cleanCusolverLUDecompositionResult(h_L, h_U, N);
        printf("Result passes verification: %d\n", verifyLUDecomposition(originalMatrix.get(), h_L, h_U, N, verbose));

        free(h_L);
        free(h_U);
    }
    printf("Total time used (s): %4.4f\n", totalTime);

    nvshmem_free(d_matrix);
    nvshmem_free(d_dependencies);
    checkCudaErrors(cudaFreeHost(h_dependencies));
    checkCudaErrors(cudaFree(d_info));
    checkCudaErrors(cudaFree(d_workspace_cusolver));
    for (int i = 0; i < workspaces; i++)
    {
        checkCudaErrors(cudaFree(d_workspace_cublas[i]));
    }
}

int main(int argc, char **argv)
{
    auto cmdl = argh::parser(argc, argv);

    if (!parseCommonArgs(cmdl, cfg)) {
        printPartitionedUsage(argv[0]);
        return 1;
    }

    if (!(cmdl({"p", "P"}, nodeToPrint) >> nodeToPrint) || nodeToPrint < -1) {
        std::cerr << "Must provide a valid node-to-print value!" << std::endl;
        return 1;
    }

    // Partitioned LU uses global PE ids (nvshmem_my_pe) rather than node-local ids.
    nvshmem_init();
    myPE = nvshmem_my_pe();
    myNodePE = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    nPE = nvshmem_n_pes();
    checkCudaErrors(cudaSetDevice(myNodePE));

    verbose = cmdl[{"v", "verbose"}] && myPE == 0;

    if (verbose)
        printf("Hello from NVSHMEM_PE=%d/%d\n", myPE, nPE);

    B = N / T;

    if (myPE == 0) {
        std::cout << "PARTITIONED LU with N=" << N << " (" << T << " of " << B << "x" << B << " tiles)" << std::endl;
        std::cout << "SM Limit per kernel = " << smLimit << std::endl;
        std::cout << "cuBLAS workspace = " << workspace << " kB" << std::endl;
    }

    if (nPE < 2) {
        std::cerr << "Partitioned LU requires multiple GPUs. Run with multiple MPI processes." << std::endl;
        return 1;
    }

    tiledLUPart(cmdl["verify"] && myPE == 0, cmdl["dot"]);

    nvshmem_finalize();
    return 0;
}