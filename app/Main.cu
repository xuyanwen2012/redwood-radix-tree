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
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    morton_keys[i] = PointToCode(inputs[i].x(), inputs[i].y(), inputs[i].z(),
                                 min_coord, range);
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void AcceleratedComputeMortonKernel(Eigen::Vector3f* inputs,
                                               Code_t* morton_keys, const int n,
                                               const float min_coord,
                                               const float range) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto x = inputs[i].x();
    const auto y = inputs[i].y();
    const auto z = inputs[i].z();
    const auto code = PointToCode(x, y, z, min_coord, range);
    morton_keys[i] = code;
  }
}

template <uint8_t Axis>
__host__ __device__ bool CompareAxis(const Eigen::Vector3f& a,
                                     const Eigen::Vector3f& b) {
  if constexpr (Axis == 0) {
    return a.x() < b.x();
  } else if constexpr (Axis == 1) {
    return a.y() < b.y();
  }
  return a.z() < b.z();
}

__global__ void CalculateEdgeCountKernel(int* edge_count,
                                         const brt::InnerNodes* inners,
                                         const int num_brt_nodes) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;

  // root has no parent, so don't do for index 0
  if (i > 0 && i < num_brt_nodes) {
    const int my_depth = inners[i].delta_node / 3;
    const int parent_depth = inners[inners[i].parent].delta_node / 3;
    edge_count[i] = my_depth - parent_depth;
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

  Code_t* u_morton_keys = nullptr;
  HANDLE_ERROR(cudaMallocManaged(&u_morton_keys, input_size * sizeof(Code_t)));

  redwood::UsmVector<Code_t> u_sorted_morton_keys(input_size);

  // [Step 1] Compute Morton Codes
  TimeTask("Compute Morton Codes", [&] {
    constexpr int block_size = 1024;
    const int num_blocks = (input_size + block_size - 1) / block_size;

    ComputeMortonKernel<<<num_blocks, block_size>>>(
        u_inputs.data(), u_morton_keys, input_size, min_coord, range);
    HANDLE_ERROR(cudaDeviceSynchronize());
  });

  // no longer need inputs
  u_inputs.clear();

  // [Step 2] Sort Morton Codes by Key
  // Enable CUB's caching allocator
  cub::CachingDeviceAllocator g_allocator(true);

  TimeTask("Sort Morton", [&] {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, u_morton_keys,
        u_sorted_morton_keys.data(), input_size));

    HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, u_morton_keys,
        u_sorted_morton_keys.data(), input_size));
    HANDLE_ERROR(cudaDeviceSynchronize());
  });

  // no longer need unsorted morton keys
  HANDLE_ERROR(cudaFree(u_morton_keys));

  // [Step 3-4] Handle Duplicates
  TimeTask("Handle Duplicates", [&] {
    u_sorted_morton_keys.erase(
        std::unique(u_sorted_morton_keys.begin(), u_sorted_morton_keys.end()),
        u_sorted_morton_keys.end());
  });

  const auto num_unique_keys = u_sorted_morton_keys.size();
  std::cout << "Actual num keys: " << num_unique_keys << '\n';

  // [Step 5] Build Binary Radix Tree
  const auto num_brt_nodes = num_unique_keys - 1;
  redwood::UsmVector<brt::InnerNodes> u_brt_nodes(num_brt_nodes);

  TimeTask("Build Binary Radix Tree", [&] {
    ProcessInternalNodes(num_unique_keys, u_sorted_morton_keys.data(),
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
    constexpr int block_size = 1024;
    const int num_blocks = (input_size + block_size - 1) / block_size;

    // the frist element is root
    u_edge_count[0] = 1;

    CalculateEdgeCountKernel<<<num_blocks, block_size>>>(
        u_edge_count.data(), u_brt_nodes.data(), num_brt_nodes);

    HANDLE_ERROR(cudaDeviceSynchronize());
  });

  // [Step 6.1] Compute Prefix Sum
  redwood::UsmVector<int> u_oc_node_offsets(num_brt_nodes + 1);
  TimeTask("Prefix Sum", [&] {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  u_edge_count.data(),
                                  u_oc_node_offsets.data() + 1, num_brt_nodes);

    HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  u_edge_count.data(),
                                  u_oc_node_offsets.data() + 1, num_brt_nodes);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // To turn this prefix sum array into a range array, we need to shift it
    u_oc_node_offsets[0] = 0;
  });

  // [Step 6.2] Allocate BH nodes
  const int num_oc_nodes = u_oc_node_offsets.back();
  const int root_level = u_brt_nodes[0].delta_node / 3;
  const Code_t root_prefix =
      u_sorted_morton_keys[0] >> (kCodeLen - (3 * root_level));
  redwood::UsmVector<oct::OctNode> u_bh_nodes(num_oc_nodes);

  // Debug print
  std::cout << "Num Unique Morton Keys: " << num_unique_keys << "\n";
  std::cout << "Num Radix Nodes: " << num_brt_nodes << "\n";
  std::cout << "Num Octree Nodes: " << num_oc_nodes << "\n";

  // [Step 7] Create unlinked BH nodes
  TimeTask("Make Unlinked BH nodes", [&] {
    MakeNodes(u_bh_nodes.data(), u_oc_node_offsets.data(), u_edge_count.data(),
              u_sorted_morton_keys.data(), u_brt_nodes.data(), num_brt_nodes,
              range);
  });

  // [Step 8] Linking BH nodes
  TimeTask("Link BH nodes", [&] {
    LinkNodes(u_bh_nodes.data(), u_oc_node_offsets.data(), u_edge_count.data(),
              u_sorted_morton_keys.data(), u_brt_nodes.data(), num_brt_nodes);
  });

  CheckTree(root_prefix, root_level * 3, u_bh_nodes.data(), 0,
            u_sorted_morton_keys.data());

  return 0;
}
