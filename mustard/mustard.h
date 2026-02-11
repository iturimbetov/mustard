#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <map>
#include <vector>
#include <utility>
#include <algorithm>

#include "utils.h"
#include "broker_queue.h"

#define FLAGS_SUBG_COUNT 0
#define FLAGS_OCCUP 4
#define MAX_TILE 4000

// ---------- CUDA Graph API compatibility (12.x vs 13.0+) ----------
// CUDA 13.0 changed signatures of several graph functions to require a
// cudaGraphEdgeData* parameter; the old overloads were removed.
// These wrappers let the code compile with both API versions.
#if CUDART_VERSION >= 13000
#define MUSTARD_cudaGraphNodeGetDependentNodes(node, out, cnt) \
    cudaGraphNodeGetDependentNodes((node), (out), nullptr, (cnt))
#define MUSTARD_cudaGraphNodeGetDependencies(node, out, cnt) \
    cudaGraphNodeGetDependencies((node), (out), nullptr, (cnt))
#define MUSTARD_cudaGraphGetEdges(g, from, to, cnt) \
    cudaGraphGetEdges((g), (from), (to), nullptr, (cnt))
#define MUSTARD_cudaGraphAddDependencies(g, from, to, cnt) \
    cudaGraphAddDependencies((g), (from), (to), nullptr, (cnt))
#define MUSTARD_cudaGraphRemoveDependencies(g, from, to, cnt) \
    cudaGraphRemoveDependencies((g), (from), (to), nullptr, (cnt))
#else
#define MUSTARD_cudaGraphNodeGetDependentNodes(node, out, cnt) \
    cudaGraphNodeGetDependentNodes((node), (out), (cnt))
#define MUSTARD_cudaGraphNodeGetDependencies(node, out, cnt) \
    cudaGraphNodeGetDependencies((node), (out), (cnt))
#define MUSTARD_cudaGraphGetEdges(g, from, to, cnt) \
    cudaGraphGetEdges((g), (from), (to), (cnt))
#define MUSTARD_cudaGraphAddDependencies(g, from, to, cnt) \
    cudaGraphAddDependencies((g), (from), (to), (cnt))
#define MUSTARD_cudaGraphRemoveDependencies(g, from, to, cnt) \
    cudaGraphRemoveDependencies((g), (from), (to), (cnt))
#endif

extern int myPE;

typedef std::pair<int, int> MatrixTile;

namespace mustard {
    
    __global__ void kernel_dep_wait(int *dependencies, int nodeIndex, int myPE)
    {
        while (nvshmem_int_atomic_fetch(dependencies + nodeIndex, myPE) > 0) { }
    }
    
    __global__ void kernel_dep_update_noq(int *dependencies, 
                                          int nodeIndex,
                                          int PE,
                                          int srcPE=-1)
    {
        nvshmem_int_atomic_fetch_add(dependencies + nodeIndex, -1, PE);
    }
    
    __global__ void kernel_dep_update(BrokerWorkDistributor queue, 
                                             int *dependencies, 
                                             int nodeIndex)
    {
        int old = nvshmem_int_atomic_fetch_add(dependencies + nodeIndex, -1, 0); // on PE 0
        if (old == 1) {
            queue.enqueue(nodeIndex, 0);
        }
    }

    // only 1 thread runs this
    // sm_count is negative when occupancy is reduced after a kernel has been completed
    __global__ void kernel_occupancy_update(int sm_count, volatile int *flags)
    {
        atomicAdd((int *)&flags[FLAGS_OCCUP], sm_count);
    }

    __global__ void kernel_populate_queue(BrokerWorkDistributor queue, int *dependencies, int totalNodes)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        for (int i = tid; i < totalNodes; i = i + gridDim.x * blockDim.x)
        {
            if (dependencies[i] == 0)
                queue.enqueue(i, 0);
        }
    }

    __global__ void kernel_test_dequeue(BrokerWorkDistributor queue)
    {
        unsigned int placeholder = UINT32_MAX;
        bool placeholder_bool = false;
        while (queue.size(0) > 0)
        {
            queue.dequeue(placeholder_bool, placeholder, 0);
            printf("Dequeued %d\n", placeholder);
        }
    }

    __global__ void kernel_scheduler(
        BrokerWorkDistributor queue,
        volatile int *flags,
        cudaGraphExec_t *subgraphs,
        int totalSubgraphs,
        int device)
    {
        unsigned int placeholder = UINT32_MAX;
        bool placeholder_bool = false;

        while (nvshmem_int_atomic_fetch((int *)&flags[FLAGS_SUBG_COUNT], 0) < totalSubgraphs)
        {
            if (flags[FLAGS_OCCUP] < (100) && queue.size(0) > 0)
            {
                queue.dequeue(placeholder_bool, placeholder, 0);
                if (placeholder_bool) {
                    cudaGraphLaunch(subgraphs[placeholder], cudaStreamGraphFireAndForget);
                    nvshmem_int_atomic_inc((int *)&flags[FLAGS_SUBG_COUNT], 0);
                }
            }
        }
    }


    class TiledGraphCreator
    {
    public:
        cudaGraph_t *subgraphs;
        cudaGraph_t graph;
        std::vector<std::vector<int>> subgraphDependencies;

        TiledGraphCreator(cudaStream_t stream, cudaGraph_t graph, bool subgraph = false, int totalNodes = 1) : stream(stream), graph(graph)
        {
            this->lastModifiedTile = std::make_pair(-1, -1);
            this->subgraph = subgraph;
            this->subgraphs = new cudaGraph_t[totalNodes];
            this->subgraphDependencies.resize(totalNodes);
            this->index_counter = 0;
        }

        void beginCaptureOperation(MatrixTile tileToWrite, std::initializer_list<MatrixTile> tilesToRead)
        {            
            auto tiles = std::vector<MatrixTile>(tilesToRead);
            tiles.push_back(tileToWrite);

            this->lastModifiedTile = tileToWrite;

            if (!this->subgraph) {
                auto dependencies = this->getDependencies(tiles);
                this->lastDependencies = dependencies;
                checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), 
                                                            nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
            } else {
                this->subgraphDependencies[index_counter] = this->getSubgraphDependencies(tiles);
                
                checkCudaErrors(cudaGraphCreate(&this->subgraphs[index_counter], 0));
                checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->subgraphs[index_counter], nullptr, 
                                                            nullptr, 0, cudaStreamCaptureModeGlobal));
            }
        }

        void endCaptureOperation()
        {
            assert(this->lastModifiedTile.first != -1 && this->lastModifiedTile.second != -1);
            if (!this->subgraph) {
                checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
                this->tileLastModifiedByMap[this->lastModifiedTile] = this->getTailOfLastCapturedNodeChain();
            } else {
                checkCudaErrors(cudaStreamEndCapture(this->stream, &(this->subgraphs[index_counter])));
                this->tileIndexByMap[this->lastModifiedTile] = this->index_counter;
                this->index_counter++;
            }
            this->lastModifiedTile = std::make_pair(-1, -1);
        };
        
        void printDeps()
        {
            for (int i = 0; i < this->subgraphDependencies.size(); i++)
            {
                std::vector<int> deps = this->subgraphDependencies[i];
                std::cout << i << ":";
                for (int j = 0; j < deps.size(); j++)
                {
                    std::cout << " " << deps[j];
                }
                std::cout << std::endl;
            }
        }

        void insertDependencyKernel(int src, int dst, BrokerWorkDistributor queue, int* d_dependencies)
        {
            cudaGraphNode_t dependencyUpdateNode;
            cudaKernelNodeParams params = {0};
            params.gridDim = dim3(1, 1, 1);
            params.blockDim = dim3(1, 1, 1);
            params.extra = NULL;
            params.func = (void *)kernel_dep_update;
            void *kernelArgs[3] = {&queue, &d_dependencies, &dst}; 
            params.kernelParams = kernelArgs;
            std::vector<cudaGraphNode_t> deps;
            deps.push_back(getTail(this->subgraphs[src]));
            checkCudaErrors(cudaGraphAddKernelNode(&dependencyUpdateNode, this->subgraphs[src], deps.data(),
                                                    deps.size(), &params));
        }

        // for inserting LU subgraphs that have to be constructed when the tile size is too big
        void insertSubgraph(cudaGraph_t getrfSubgraph) 
        {
            cudaGraph_t g = this->subgraphs[index_counter-1];
            std::vector<cudaGraphNode_t> deps;

            cudaGraphNode_t root, tail; 
            root = getRoot(g);
            tail = getTail(g);

            if (myPE != 0) {
                size_t edge_count;
                checkCudaErrors(MUSTARD_cudaGraphNodeGetDependentNodes(root, NULL, &edge_count));
                if (edge_count > 0) { 
                    std::vector<cudaGraphNode_t> children(edge_count);
                    checkCudaErrors(MUSTARD_cudaGraphNodeGetDependentNodes(root, children.data(), &edge_count));
                    root = children[0];
                }

                checkCudaErrors(MUSTARD_cudaGraphNodeGetDependencies(tail, NULL, &edge_count));
                if (edge_count > 0) { 
                    std::vector<cudaGraphNode_t> parents(edge_count);
                    checkCudaErrors(MUSTARD_cudaGraphNodeGetDependencies(tail, parents.data(), &edge_count));
                    tail = parents[0];
                }
            }
            deps.push_back(root);
            
            cudaGraphNode_t node;
            checkCudaErrors(cudaGraphAddChildGraphNode(&node, g, deps.data(),
                                                        deps.size(), getrfSubgraph));
            //if (myPE == 0) {
                MUSTARD_cudaGraphAddDependencies(g, &node, &tail, 1); // add dep from subg to write memcpy 
                MUSTARD_cudaGraphRemoveDependencies(g, &root, &tail, 1); // remove dep from read memcpy to write memcpy
            //}
        }


    private:
        std::map<MatrixTile, cudaGraphNode_t> tileLastModifiedByMap;
        std::map<MatrixTile, int> tileIndexByMap;
        std::map<cudaGraphNode_t, bool> visited;
        cudaStream_t stream;
        MatrixTile lastModifiedTile;
        std::vector<cudaGraphNode_t> lastDependencies;
        int index_counter;
        bool subgraph;

        std::vector<cudaGraphNode_t> getDependencies(std::vector<MatrixTile> tiles)
        {
            std::vector<cudaGraphNode_t> dependencies;
            for (auto tile : tiles)
            {
                auto it = this->tileLastModifiedByMap.find(tile);
                if (it != this->tileLastModifiedByMap.end())
                {
                    dependencies.push_back(it->second);
                }
            }

            auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
            dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

            return dependencies;
        }

        std::vector<int> getSubgraphDependencies(std::vector<MatrixTile> tiles)
        {
            std::vector<int> dependencies;
            for (auto tile : tiles)
            {
                auto it = this->tileIndexByMap.find(tile);
                if (it != this->tileIndexByMap.end())
                {
                    dependencies.push_back(it->second);
                }
            }
 
            std::sort(dependencies.begin(), dependencies.end());
            auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
            dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));

            return dependencies;
        }

        cudaGraphNode_t getRoot(cudaGraph_t graph){
            size_t numNodes = 1;
            auto nodes = std::make_unique<cudaGraphNode_t[]>(1);
            checkCudaErrors(cudaGraphGetRootNodes(graph, nodes.get(), &numNodes));
            return nodes[0];
        }

        cudaGraphNode_t getTail(cudaGraph_t graph){
            size_t numEdges;
            checkCudaErrors(MUSTARD_cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges));
            if (numEdges == 0) 
            {
                size_t numNodes = 1;
                auto nodes = std::make_unique<cudaGraphNode_t[]>(1);
                checkCudaErrors(cudaGraphGetNodes(graph, nodes.get(), &numNodes));
                return nodes[0];
            }
            auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
            auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
            checkCudaErrors(MUSTARD_cudaGraphGetEdges(graph, from.get(), to.get(), &numEdges));

            std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
            std::set<cudaGraphNode_t> noOutGoingEdgeNodes;
            for (int i = 0; i < numEdges; i++)
            {
                hasOutGoingEdge[from[i]] = true;
                noOutGoingEdgeNodes.erase(from[i]);
                if (!hasOutGoingEdge[to[i]])
                    noOutGoingEdgeNodes.insert(to[i]);
            }

            if (noOutGoingEdgeNodes.size() != 1) {
                size_t numNodes = 0;
                checkCudaErrors(cudaGraphGetNodes(graph, nullptr, &numNodes));
            
                std::vector<cudaGraphNode_t> nodes(numNodes);
                cudaGraphGetNodes(graph, nodes.data(), &numNodes);
                return nodes.back();
            }

            return *noOutGoingEdgeNodes.rbegin();
        }

        cudaGraphNode_t getTailOfLastCapturedNodeChain()
        {
            if (lastDependencies.size() == 0)
            {
                return getTail(this->graph);
            }
            else
            {
                auto nodeBeforeChain = lastDependencies[0];
                size_t numDependentNodes;
                checkCudaErrors(MUSTARD_cudaGraphNodeGetDependentNodes(nodeBeforeChain, nullptr, &numDependentNodes));

                assert(numDependentNodes > 0);

                auto dependentNodes = std::make_unique<cudaGraphNode_t[]>(numDependentNodes);
                checkCudaErrors(MUSTARD_cudaGraphNodeGetDependentNodes(nodeBeforeChain, dependentNodes.get(), &numDependentNodes));

                cudaGraphNode_t chainBeginningNode;
                for (int i = 0; i < numDependentNodes; i++)
                {
                    if (!visited[dependentNodes[i]])
                    {
                        chainBeginningNode = dependentNodes[i];
                        break;
                    }
                }

                auto u = chainBeginningNode;
                while (true)
                {
                    visited[u] = true;
                    checkCudaErrors(MUSTARD_cudaGraphNodeGetDependentNodes(u, nullptr, &numDependentNodes));
                    if (numDependentNodes == 0)
                        break;

                    std::vector<cudaGraphNode_t> nodes(numDependentNodes);
                    checkCudaErrors(MUSTARD_cudaGraphNodeGetDependentNodes(u, nodes.data(), &numDependentNodes));
                    u = nodes.back();
                }

                return u;
            }
        }
    };

}