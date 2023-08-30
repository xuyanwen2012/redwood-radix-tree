#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <numeric>

#include "Morton.hpp"

namespace cg = cooperative_groups;

__global__ void ComputeMortonKernel(Eigen::Vector3f* inputs,
                                    Code_t* morton_keys, const int n,
                                    const float min_coord, const float range) {
  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  if (tid < n) {
    morton_keys[tid] = PointToCode(inputs[tid].x(), inputs[tid].y(),
                                   inputs[tid].z(), min_coord, range);
  }
}

// void foo(const float *A, const float *B, float *C, int rowsA, int colsA,
//          int colsB) {
//   dim3 threadsPerBlock(16, 16);
//   dim3 numBlocks((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                  (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);

//   float *d_A, *d_B, *d_C;  // Device memory pointers

//   // Allocate device memory for matrices A, B, and C
//   cudaMalloc((void **)&d_A, rowsA * colsA * sizeof(float));
//   cudaMalloc((void **)&d_B, colsA * colsB * sizeof(float));
//   cudaMalloc((void **)&d_C, rowsA * colsB * sizeof(float));

//   // Copy data from host to device
//   cudaMemcpy(d_A, A, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_B, B, colsA * colsB * sizeof(float), cudaMemcpyHostToDevice);

//   // Launch the matrix multiplication kernel
//   matrixMultKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rowsA,
//   colsA,
//                                                    colsB);

//   // Copy the result matrix C from device to host
//   cudaMemcpy(C, d_C, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

//   // Free device memory
//   cudaFree(d_A);
//   cudaFree(d_B);
//   cudaFree(d_C);

//   cudaDeviceSynchronize();  // Wait for the kernel and memory copies to
//   finish
// }

int main() {
  thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Prepare Inputs
  constexpr int input_size = 1024;

  auto unifed_mem_used = 0;

  Eigen::Vector3f* u_inputs = nullptr;

  HANDLE_ERROR(
      cudaMallocManaged(&u_inputs, input_size * sizeof(Eigen::Vector3f)));
  unifed_mem_used += input_size * sizeof(Eigen::Vector3f);

  float min_coord = 0;
  float range = 1024.0f;
  std::generate_n(u_inputs, input_size, [&] {
    const auto x = dis(gen) * 1024.0f;
    const auto y = dis(gen) * 1024.0f;
    const auto z = dis(gen) * 1024.0f;
    return Eigen::Vector3f(x, y, z);
  });

  std::cout << "Unified Memory Used: " << unifed_mem_used << " bytes\n";

  std::cout << "Peek Input\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << u_inputs[i].transpose() << '\n';
  }

  Code_t* u_morton_keys = nullptr;
  HANDLE_ERROR(cudaMallocManaged(&u_morton_keys, input_size * sizeof(Code_t)));
  unifed_mem_used += input_size * sizeof(Code_t);

  // ComputeMortonKernel<<<1, 1024>>>(u_inputs, u_morton_keys, input_size,
                                  //  min_coord, range);

  // HANDLE_ERROR(cudaDeviceSynchronize());

  std::vector<Code_t> morton_keys;
  std::transform(u_inputs, u_inputs + input_size, std::back_inserter(morton_keys),
                 [&](const auto& vec) {
                   return PointToCode(vec.x(), vec.y(), vec.z(), min_coord,
                                      range);
                 });

  std::cout << "Peek Morton Keys\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << u_morton_keys[i] << '\n';
  }

  // no longer need inputs
  cudaFree(u_inputs);

  // morton_keys.erase(std::unique(morton_keys, morton_keys + input_size),
  //                   morton_keys.end());

  cudaFree(u_morton_keys);

  return 0;
}
