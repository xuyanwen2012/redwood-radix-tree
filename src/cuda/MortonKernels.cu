#include <cooperative_groups.h>

#include <cub/cub.cuh>

#include "MortonKernels.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace cg = cooperative_groups;

__global__ void ComputeMortonKernel(const Eigen::Vector3f* inputs,
                                    Code_t* morton_keys, const size_t n,
                                    const float min_coord, const float range) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    morton_keys[i] = PointToCode(inputs[i].x(), inputs[i].y(), inputs[i].z(),
                                 min_coord, range);
  }
}

void ComputeMortonCodes(const Eigen::Vector3f* inputs, Code_t* morton_keys,
                        const size_t n, const float min_coord,
                        const float range) {
  const auto num_blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
  ComputeMortonKernel<<<num_blocks, kThreadsPerBlock>>>(inputs, morton_keys, n,
                                                        min_coord, range);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void SortMortonCodes(const Code_t* morton_keys, Code_t* sorted_morton_keys,
                     const size_t n) {
  // Enable CUB's caching allocator
  cub::CachingDeviceAllocator g_allocator(true);

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, morton_keys, sorted_morton_keys, n));

  HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, morton_keys, sorted_morton_keys, n));

  HANDLE_ERROR(cudaDeviceSynchronize());
}