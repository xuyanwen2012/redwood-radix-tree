#include <Eigen/Dense>
#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

#include "BinaryRadixTree.hpp"
#include "MortonKernels.hpp"
#include "Octree.hpp"
#include "PrefixSum.hpp"
#include "UnifiedSharedMemory.hpp"
#include "Utils.hpp"

template <uint8_t Axis>
bool CompareAxis(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  if constexpr (Axis == 0) {
    return a.x() < b.x();
  } else if constexpr (Axis == 1) {
    return a.y() < b.y();
  }
  return a.z() < b.z();
}

int main() {
  thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  constexpr auto input_size = 640 * 480;

  redwood::UsmVector<Eigen::Vector3f> u_inputs(input_size);
  redwood::UsmVector<Code_t> u_morton_keys(input_size);
  redwood::UsmVector<Code_t> u_sorted_morton_keys(input_size);

  std::generate(u_inputs.begin(), u_inputs.end(), [&] {
    const auto x = dis(gen);  //* 1024.0f;
    const auto y = dis(gen);  //* 1024.0f;
    const auto z = dis(gen);  //* 1024.0f;
    return Eigen::Vector3f(x, y, z);
  });

  float min_coord = 0.0f;
  float max_coord = 1.0f;

  TimeTask("Find Min Max", [&] {
    const auto x_range =
        std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<0>);
    const auto y_range =
        std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<1>);
    const auto z_range =
        std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<2>);
    const auto x_min = x_range.first;
    const auto x_max = x_range.second;
    const auto y_min = y_range.first;
    const auto y_max = y_range.second;
    const auto z_min = z_range.first;
    const auto z_max = z_range.second;
    std::array<float, 3> mins{x_min->x(), y_min->y(), z_min->z()};
    std::array<float, 3> maxes{x_max->x(), y_max->y(), z_max->z()};
    min_coord = *std::min_element(mins.begin(), mins.end());
    max_coord = *std::max_element(maxes.begin(), maxes.end());
  });

  const float range = max_coord - min_coord;

  std::cout << "Min: " << min_coord << "\n";
  std::cout << "Max: " << max_coord << "\n";
  std::cout << "Range: " << range << "\n";

  TimeTask("[Step 1]: Compute Morton Codes", [&] {
    ComputeMortonCodes(u_inputs.data(), u_morton_keys.data(), input_size,
                       min_coord, range);
  });

  TimeTask("[Step 2]: Sort Morton Codes", [&] {
    SortMortonCodes(u_morton_keys.data(), u_sorted_morton_keys.data(),
                    input_size);
  });

  redwood::UsmVector<Eigen::Vector3f>().swap(u_inputs);
  redwood::UsmVector<Code_t>().swap(u_morton_keys);

  TimeTask("Handle Duplicates", [&] {
    u_sorted_morton_keys.erase(
        std::unique(u_sorted_morton_keys.begin(), u_sorted_morton_keys.end()),
        u_sorted_morton_keys.end());
  });

  const auto num_unique_keys = u_sorted_morton_keys.size();
  const auto num_brt_nodes = num_unique_keys - 1;
  std::cout << "Actual num keys: " << num_unique_keys << '\n';

  redwood::UsmVector<brt::InnerNodes> u_brt_nodes(num_brt_nodes);

  TimeTask("[Step 3]: Build Radix Tree", [&] {
    BuildBinaryRadixTree(u_sorted_morton_keys.data(), num_unique_keys,
                         u_brt_nodes.data());
  });

  for (int i = 0; i < num_brt_nodes; ++i) {
    std::cout << "Node " << i << ": " << u_brt_nodes[i].delta_node << ", "
              << u_brt_nodes[i].left << ", " << u_brt_nodes[i].right << "\n";
  }

  redwood::UsmVector<int> u_edge_count(num_brt_nodes);
  redwood::UsmVector<int> u_oc_node_offsets(num_brt_nodes + 1);

  std::cout << "num_brt_nodes: " << num_brt_nodes;
  std::cout << "num_brt_nodes: " << num_brt_nodes + 1;

  TimeTask("[Step 4]: Count Edges", [&] {
    CalculateEdgeCount(u_brt_nodes.data(), u_edge_count.data(), num_brt_nodes);
  });

  TimeTask("[Step 5]: Prefix Sum", [&] {
    ComputeRangeArray(u_edge_count.data(), u_oc_node_offsets.data(),
                      num_brt_nodes);
  });

  const int num_oc_nodes = u_oc_node_offsets.back();
  const int root_level = u_brt_nodes[0].delta_node / 3;
  const Code_t root_prefix =
      u_sorted_morton_keys[0] >> (kCodeLen - (3 * root_level));
  redwood::UsmVector<oct::OctNode> u_bh_nodes(num_oc_nodes);

  // Debug print
  std::cout << "Num Unique Morton Keys: " << num_unique_keys << "\n";
  std::cout << "Num Radix Nodes: " << num_brt_nodes << "\n";
  std::cout << "Num Octree Nodes: " << num_oc_nodes << "\n";

  TimeTask("[Step 6]: Make Octree Nodes", [&] {
    MakeUnlinkedOctreeNodes(u_bh_nodes.data(), u_oc_node_offsets.data(),
                            u_edge_count.data(), u_sorted_morton_keys.data(),
                            u_brt_nodes.data(), num_brt_nodes, min_coord,
                            range);
  });

  TimeTask("[Step 7]: Link Nodes", [&] {
    LinkOctreeNodes(u_bh_nodes.data(), u_oc_node_offsets.data(),
                    u_edge_count.data(), u_sorted_morton_keys.data(),
                    u_brt_nodes.data(), num_brt_nodes);
  });

  CheckTree(root_prefix, root_level * 3, u_bh_nodes.data(), 0,
            u_sorted_morton_keys.data());

  return 0;
}
