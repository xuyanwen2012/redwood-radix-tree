#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

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

  ComputeMortonKernel<<<1, 1024>>>(u_inputs, u_morton_keys, input_size,
                                   min_coord, range);

  HANDLE_ERROR(cudaDeviceSynchronize());

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
