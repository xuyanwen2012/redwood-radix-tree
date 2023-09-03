#include <cub/cub.cuh>

#include "PrefixSum.hpp"
#include "cuda/CudaUtils.cuh"

__device__ void CalculateEdgeCountHelper(const int i, int* edge_count,
                                         const brt::InnerNodes* inners) {
  const int my_depth = inners[i].delta_node / 3;
  const int parent_depth = inners[inners[i].parent].delta_node / 3;
  edge_count[i] = my_depth - parent_depth;
}

__global__ void CalculateEdgeCountKernel(
    int* edge_count, const brt::InnerNodes* __restrict__ inners,
    const int num_brt_nodes) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  // root has no parent, so don't do for index 0
  if (i > 0 && i < num_brt_nodes) {
    CalculateEdgeCountHelper(i, edge_count, inners);
  }
}

void CalculateEdgeCount(const brt::InnerNodes* brt_nodes, int* edge_count,
                        const size_t num_nodes) {
  // the frist element is root
  edge_count[0] = 1;

  const auto num_blocks =
      (num_nodes + kThreadsPerBlock - 1) / kThreadsPerBlock;  // round up
  CalculateEdgeCountKernel<<<num_blocks, kThreadsPerBlock>>>(
      edge_count, brt_nodes, num_nodes);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void PrefixSum(const int* input, int* output, const size_t n) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, input,
                                output, n);

  HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, input,
                                output, n);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void ComputeRangeArray(const int* edge_count, int* node_offset,
                       const size_t num_nodes) {
  // masure node_offset size == edge_count size + 1
  PrefixSum(edge_count, node_offset + 1, num_nodes);
  // To turn this prefix sum array into a range array, we need to shift it
  node_offset[0] = 0;
}