#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdlib>
#include <cub/cub.cuh>
#include <iostream>
#include <numeric>
#include <random>

#include "BinaryRadixTree.hpp"
#include "Morton.hpp"
#include "Octree.hpp"
#include "UnifiedSharedMemory.hpp"

namespace cg = cooperative_groups;

__global__ void ComputeMortonKernel(Eigen::Vector3f* inputs,
                                    Code_t* morton_keys, const int n,
                                    const float min_coord, const float range) {
  // auto cta = cg::this_thread_block();
  // const auto tid = cta.thread_rank();

  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    morton_keys[i] = PointToCode(inputs[i].x(), inputs[i].y(), inputs[i].z(),
                                 min_coord, range);
  }
}

template <uint8_t Axis>
__host__ __device__ bool CompareAxis(const Eigen::Vector3f& a,
                                     const Eigen::Vector3f& b) {
  if constexpr (Axis == 0) {
    return a.x() < b.x();
  } else if constexpr (Axis == 1) {
    return a.y() < b.y();
  } else {
    return a.z() < b.z();
  }
}

int main() {
  thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Prepare Inputs
  // constexpr int input_size = 640 * 480;
  constexpr int input_size = 1280 * 720;

  redwood::UsmVector<Eigen::Vector3f> u_inputs(input_size);

  std::generate(u_inputs.begin(), u_inputs.end(), [&] {
    const auto x = dis(gen) * 1024.0f;
    const auto y = dis(gen) * 1024.0f;
    const auto z = dis(gen) * 1024.0f;
    return Eigen::Vector3f(x, y, z);
  });

  float min_coord = 0.0f;
  float max_coord = 1.0f;

  TimeTask("Find Min Max", [&] {
    auto x_range =
        std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<0>);
    auto y_range =
        std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<1>);
    auto z_range =
        std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<2>);
    auto x_min = x_range.first;
    auto x_max = x_range.second;
    auto y_min = y_range.first;
    auto y_max = y_range.second;
    auto z_min = z_range.first;
    auto z_max = z_range.second;
    std::array<float, 3> mins{x_min->x(), y_min->y(), z_min->z()};
    std::array<float, 3> maxes{x_max->x(), y_max->y(), z_max->z()};
    min_coord = *std::min_element(mins.begin(), mins.end());
    max_coord = *std::max_element(maxes.begin(), maxes.end());
  });

  const float range = max_coord - min_coord;

  std::cout << "Min: " << min_coord << "\n";
  std::cout << "Max: " << max_coord << "\n";
  std::cout << "Range: " << range << "\n";

  // std::cout << "Peek Input\n";
  // for (int i = 0; i < 5; ++i) {
  //   std::cout << u_inputs[i].transpose() << '\n';
  // }

  Code_t* u_morton_keys = nullptr;
  HANDLE_ERROR(cudaMallocManaged(&u_morton_keys, input_size * sizeof(Code_t)));

  redwood::UsmVector<Code_t> sorted_morton_keys(input_size);

  // [Step 1] Compute Morton Codes
  TimeTask("Compute Morton Codes", [&] {
    const int block_size = 1024;
    const int num_blocks = (input_size + block_size - 1) / block_size;

    ComputeMortonKernel<<<num_blocks, block_size>>>(
        u_inputs.data(), u_morton_keys, input_size, min_coord, range);
    HANDLE_ERROR(cudaDeviceSynchronize());
  });

  const auto num_keys = input_size;

  // no longer need inputs
  u_inputs.clear();

  // [Step 2] Sort Morton Codes by Key

  cub::CachingDeviceAllocator g_allocator(
      true);  // Enable CUB's caching allocator
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  TimeTask("Sort Morton", [&] {
    HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, u_morton_keys,
        sorted_morton_keys.data(), num_keys));

    HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, u_morton_keys,
        sorted_morton_keys.data(), num_keys));
    HANDLE_ERROR(cudaDeviceSynchronize());
  });

  cudaFree(u_morton_keys);
  // [Step 3-4] Handle Duplicates
  TimeTask("Handle Duplicates", [&] {
    sorted_morton_keys.erase(
        std::unique(sorted_morton_keys.begin(), sorted_morton_keys.end()),
        sorted_morton_keys.end());
  });

  const auto num_unique_keys = sorted_morton_keys.size();
  std::cout << "Actual num keys: " << num_unique_keys << '\n';

  const auto num_brt_nodes = num_unique_keys - 1;

  // [Step 5] Build Binary Radix Tree
  redwood::UsmVector<brt::InnerNodes> u_brt_nodes(num_brt_nodes);

  TimeTask("Build Binary Radix Tree", [&] {
    ProcessInternalNodes(num_unique_keys, sorted_morton_keys.data(),
                         u_brt_nodes.data());
  });

  if (false) {
    for (int i = 0; i < num_brt_nodes; ++i) {
      std::cout << "Node " << i << "\n";
      std::cout << "\tdelta_node: " << u_brt_nodes[i].delta_node << "\n";
      std::cout << "\tleft: " << u_brt_nodes[i].left << "\n";
      std::cout << "\tright: " << u_brt_nodes[i].right << "\n";
      std::cout << "\tparent: " << u_brt_nodes[i].parent << "\n";
      std::cout << "\n";
    }
  }

  // [Step 6] Count edges
  redwood::UsmVector<int> u_edge_count(num_brt_nodes);
  TimeTask("Count Edges", [&] {
    // Copy a "1" to the first element to account for the root
    u_edge_count[0] = 1;
    oct::CalculateEdgeCount(u_edge_count.data(), u_brt_nodes.data(),
                            num_brt_nodes);
  });

  // [Step 6.1] Prefix sum
  redwood::UsmVector<int> u_oc_node_offsets(num_brt_nodes + 1);
  TimeTask("Prefix Sum", [&] {
    std::partial_sum(u_edge_count.begin(), u_edge_count.end(),
                     u_oc_node_offsets.begin() + 1);
    u_oc_node_offsets[0] = 0;
  });

  // [Step 6.2] Allocate BH nodes
  const int num_oc_nodes = u_oc_node_offsets.back();
  const int root_level = u_brt_nodes[0].delta_node / 3;
  const Code_t root_prefix = u_morton_keys[0] >> (kCodeLen - (3 * root_level));
  redwood::UsmVector<oct::OctNode> bh_nodes(num_oc_nodes);

  // Debug print
  std::cout << "Num Unique Morton Keys: " << num_unique_keys << "\n";
  std::cout << "Num Radix Nodes: " << num_brt_nodes << "\n";
  std::cout << "Num Octree Nodes: " << num_oc_nodes << "\n";

  // // [Step 7] Create unlinked BH nodes
  // TimeTask("Make Unlinked BH nodes", [&] {
  //   MakeNodes(bh_nodes.data(), u_oc_node_offsets.data(), u_edge_count.data(),
  //             u_morton_keys.data(), u_brt_nodes.data(), num_brt_nodes, range);
  // });

  return 0;
}
