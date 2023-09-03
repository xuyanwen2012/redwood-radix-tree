#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "BinaryRadixTree.hpp"
#include "cuda/CudaUtils.cuh"

namespace cg = cooperative_groups;

namespace brt {

__global__ void BuildBinaryRadixTreeKernel(const Code_t* morton_keys,
                                           const size_t key_num,
                                           InnerNodes* brt_nodes) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  const auto num_brt_nodes = key_num - 1;
  if (i < num_brt_nodes) {
    ProcessInternalNodesHelper(key_num, morton_keys, i, brt_nodes);
  }
}

}  // namespace brt

void BuildBinaryRadixTree(const Code_t* sorted_morton,
                          const size_t num_unique_keys,
                          brt::InnerNodes* brt_nodes) {
  const auto num_brt_nodes = num_unique_keys - 1;

  const auto num_blocks =
      (num_brt_nodes + kThreadsPerBlock - 1) / kThreadsPerBlock;
  BuildBinaryRadixTreeKernel<<<num_blocks, kThreadsPerBlock>>>(
      sorted_morton, num_unique_keys, brt_nodes);
  HANDLE_ERROR(cudaDeviceSynchronize());
}
