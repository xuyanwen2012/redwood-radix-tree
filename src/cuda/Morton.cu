#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "Morton.hpp"
#include "cuda/CudaUtils.cuh"

namespace cg = cooperative_groups;

__host__ __device__ Code_t PointToCode(const float x, const float y,
                                       const float z, const float min_coord,
                                       const float range) {
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  const auto x_coord =
      static_cast<uint32_t>(bit_scale * ((x - min_coord) / range));
  const auto y_coord =
      static_cast<uint32_t>(bit_scale * ((y - min_coord) / range));
  const auto z_coord =
      static_cast<uint32_t>(bit_scale * ((z - min_coord) / range));

  return Encode64(x_coord, y_coord, z_coord);
}

__host__ __device__ Eigen::Vector3f CodeToPoint(const Code_t code,
                                                const float min_coord,
                                                const float range) {
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  uint32_t dec_raw_x, dec_raw_y, dec_raw_z;
  Decode64(code, dec_raw_x, dec_raw_y, dec_raw_z);
  float dec_x = (static_cast<float>(dec_raw_x) / bit_scale) * range + min_coord;
  float dec_y = (static_cast<float>(dec_raw_y) / bit_scale) * range + min_coord;
  float dec_z = (static_cast<float>(dec_raw_z) / bit_scale) * range + min_coord;
  return {dec_x, dec_y, dec_z};
}

__global__ void ComputeMortonKernel(const Eigen::Vector3f* inputs,
                                    Code_t* morton_keys, const int n,
                                    const float min_coord, const float range) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    morton_keys[i] = PointToCode(inputs[i].x(), inputs[i].y(), inputs[i].z(),
                                 min_coord, range);
  }
}

void ComputeMortonCodes(const Eigen::Vector3f* inputs, Code_t* morton_keys,
                        const int n, const float min_coord, const float range) {
  constexpr int block_size = 1024;
  const int num_blocks = (n + block_size - 1) / block_size;
  ComputeMortonKernel<<<num_blocks, block_size>>>(inputs, morton_keys, n,
                                                  min_coord, range);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void SortMortonCodes(const Code_t* morton_keys, Code_t* sorted_morton_keys,
                     int n) {
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