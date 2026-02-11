#pragma once

#include <iostream>
#include <string>
#include "argh.h"
#include "utils.h"

// Common configuration shared by all mustard executables.
struct MustardConfig {
    size_t N = 15;
    size_t B = 3;
    size_t T = 5;
    int    myPE = 0;
    int    verbose = 0;
    int    workspace = 256;   // cuBLAS workspace in kB
    int    smLimit = 20;
    int    runs = 1;
};

// Print common options shared by all executables.
inline void printCommonUsage()
{
    std::cerr << "\n  Common options:\n"
              << "    -n, -N=<int>         Matrix dimension N                       [default: 15]\n"
              << "    -t, -T=<int>         Number of tiles (N must be divisible)    [default: 5]\n"
              << "    --sm, --smLimit=<int> SM limit per kernel (1-108)             [default: 20]\n"
              << "    --ws, --workspace=<int> cuBLAS workspace in kB (1-1048576)   [default: 256]\n"
              << "    -r, --runs=<int>     Number of timing runs                    [default: 1]\n"
              << "    -v, --verbose        Enable verbose output\n"
              << "    --verify             Verify result correctness\n"
              << "    --dot                Dump execution graph in DOT format\n";
}

// Print usage for lu_mustard / cholesky_mustard (single-node).
inline void printSingleNodeUsage(const char* progName, const char* decomposition)
{
    std::cerr << "Usage: " << progName << " [options]\n"
              << "\n  " << decomposition << " decomposition on one or more GPUs using CUDA graphs.\n"
              << "  The number of GPUs is determined by the number of NVSHMEM PEs (MPI ranks).\n"
              << "\n  Mode (pick one; default is single-kernel if none given):\n"
              << "    --tiled              Tiled execution (one graph per tile step)\n"
              << "    --subgraph           Sub-graph (mustard) execution\n";
    printCommonUsage();
    std::cerr << "\n  Examples:\n"
              << "    " << progName << " -n=600 -t=2 --tiled --verify\n"
              << "    nvshmrun -np 4 " << progName << " -n=6000 -t=10 --subgraph -r=5\n"
              << std::endl;
}

// Print usage for p_lu_mustard (partitioned multi-node LU).
inline void printPartitionedUsage(const char* progName)
{
    std::cerr << "Usage: " << progName << " [options]\n"
              << "\n  Partitioned LU decomposition across multiple GPUs / nodes.\n"
              << "  Requires at least 2 MPI ranks (NVSHMEM PEs).\n"
              << "\n  Additional options:\n"
              << "    -p, -P=<int>         PE index whose graph to print (-1=none) [default: 0]\n";
    printCommonUsage();
    std::cerr << "\n  Examples:\n"
              << "    nvshmrun -np 4 " << progName << " -n=6000 -t=12 --verify\n"
              << "    mpirun -np 8 " << progName << " -n=12000 -t=24 -r=5\n"
              << std::endl;
}

// Parse the common CLI arguments shared by all mustard executables.
// Returns false if validation fails (error already printed).
inline bool parseCommonArgs(argh::parser& cmdl, MustardConfig& cfg)
{
    if (cmdl[{"h", "help"}]) {
        return false;  // caller checks and prints usage
    }
    if (!(cmdl({"N", "n"}, cfg.N) >> cfg.N)) {
        std::cerr << "Error: Must provide a valid N value! Got '" << cmdl({"N", "n"}).str() << "'" << std::endl;
        return false;
    }
    if (!(cmdl({"t", "T"}, cfg.T) >> cfg.T)) {
        std::cerr << "Error: Must provide a valid T value! Got '" << cmdl({"T", "t"}).str() << "'" << std::endl;
        return false;
    }
    if (cfg.N % cfg.T > 0) {
        std::cerr << "Error: N must be divisible by T! Got 'N=" << cfg.N << " & T=" << cfg.T << "'" << std::endl;
        return false;
    }
    if (!(cmdl({"sm", "SM", "smLimit"}, cfg.smLimit) >> cfg.smLimit) || cfg.smLimit > 108 || cfg.smLimit < 1) {
        std::cerr << "Error: Must provide a valid SM Limit value! Got '" << cmdl({"sm", "SM", "smLimit"}).str() << "'" << std::endl;
        return false;
    }
    if (!(cmdl({"workspace", "ws", "w", "W"}, cfg.workspace) >> cfg.workspace) || cfg.workspace > 1024 * 1024 || cfg.workspace < 1) {
        std::cerr << "Error: Must provide a valid workspace (in kBytes) value! Got '" << cmdl({"workspace", "ws", "w"}).str() << "'" << std::endl;
        return false;
    }
    if (!(cmdl({"run", "runs", "r", "R"}, cfg.runs) >> cfg.runs) || cfg.runs < 1) {
        std::cerr << "Error: Must provide a valid number of runs! Got '" << cmdl({"run", "r", "R"}).str() << "'" << std::endl;
        return false;
    }
    return true;
}

// Initialize NVSHMEM and set the CUDA device.
// Sets cfg.myPE and cfg.verbose based on command-line flags.
inline bool initNvshmemDevice(argh::parser& cmdl, MustardConfig& cfg)
{
    nvshmem_init();
    cfg.myPE = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    checkCudaErrors(cudaSetDevice(cfg.myPE));

    cfg.verbose = cmdl[{"v", "verbose"}] && cfg.myPE == 0;

    if (cfg.verbose) {
        int gpusAvailable = -1;
        checkCudaErrors(cudaGetDeviceCount(&gpusAvailable));
        printf("Hello from NVSHMEM_PE=%d/%d\n", cfg.myPE, nvshmem_n_pes());
        printf("%d GPUs detected, asked to use %d GPUs\n", gpusAvailable, nvshmem_n_pes());
    }
    return true;
}
