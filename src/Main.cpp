#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <bitset>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

#include "BinaryRadixTree.hpp"
#include "Morton.hpp"
#include "Octree.hpp"
#include "Utils.hpp"

void PrintVector3F(const Eigen::Vector3f& vec) {
  std::cout << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")\n";
}

template <uint8_t Axis>
bool CompareAxis(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
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
  static std::uniform_real_distribution dis(0.0f, 1.0f);

  // Prepare Inputs
  constexpr int input_size = 1024 * 40;
  // constexpr int input_size = 1280 * 720;
  std::vector<Eigen::Vector3f> inputs(input_size);
  std::generate(inputs.begin(), inputs.end(), [&] {
    const auto x = dis(gen) * 1024.0f;
    const auto y = dis(gen) * 1024.0f;
    const auto z = dis(gen) * 1024.0f;
    return Eigen::Vector3f(x, y, z);
  });

  float min_coord = 0.0f;
  float max_coord = 1.0f;
  TimeTask("Find Min Max", [&] {
    const auto x =
        std::minmax_element(inputs.begin(), inputs.end(), CompareAxis<0>);
    const auto y =
        std::minmax_element(inputs.begin(), inputs.end(), CompareAxis<1>);
    const auto z =
        std::minmax_element(inputs.begin(), inputs.end(), CompareAxis<2>);
    std::array<float, 3> mins{x.first->x(), y.first->y(), z.first->z()};
    std::array<float, 3> maxes{x.second->x(), y.second->y(), z.second->z()};
    min_coord = *std::min_element(mins.begin(), mins.end());
    max_coord = *std::max_element(maxes.begin(), maxes.end());
  });

  float range = max_coord - min_coord;

  std::cout << "Min: " << min_coord << "\n";
  std::cout << "Max: " << max_coord << "\n";
  std::cout << "Range: " << range << "\n";

  // [Step 1] Compute Morton Codes
  std::vector<Code_t> morton_keys;
  morton_keys.reserve(input_size);

  TimeTask("Compute Morton Codes", [&] {
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(morton_keys), [&](const auto& vec) {
                     return PointToCode(vec.x(), vec.y(), vec.z(), min_coord,
                                        range);
                   });
  });

  // [Step 2] Sort Morton Codes by Key
  TimeTask("Sort Morton Codes",
           [&] { std::sort(morton_keys.begin(), morton_keys.end()); });

  // [Step 3-4] Handle Duplicates
  TimeTask("Handle Duplicates", [&] {
    morton_keys.erase(std::unique(morton_keys.begin(), morton_keys.end()),
                      morton_keys.end());
  });

  // std::for_each(morton_keys.begin(), morton_keys.end(),
  //               [min_coord, range](const auto key) {
  //                 std::cout << key << "\t" << std::bitset<CODE_LEN>(key) <<
  //                 "\t"
  //                           << CodeToPoint(key, min_coord, range).transpose()
  //                           << std::endl;
  //               });

  // [Step 5] Build Binary Radix Tree
  const auto num_brt_nodes = morton_keys.size() - 1;
  std::vector<brt::InnerNodes> inners(num_brt_nodes);

  TimeTask("Build Binary Radix Tree", [&] {
    brt::ProcessInternalNodes(morton_keys.size(), morton_keys.data(),
                              inners.data());
  });

  // for (int i = 0; i < num_brt_nodes; ++i) {
  //   std::cout << "Node " << i << "\n";
  //   std::cout << "\tdelta_node: " << inners[i].delta_node << "\n";
  //   std::cout << "\tleft: " << inners[i].left << "\n";
  //   std::cout << "\tright: " << inners[i].right << "\n";
  //   std::cout << "\tparent: " << inners[i].parent << "\n";
  //   std::cout << "\n";
  // }

  // [Step 6] Count edges
  std::vector<int> edge_count(num_brt_nodes);
  TimeTask("Count Edges", [&] {
    // Copy a "1" to the first element to account for the root
    edge_count[0] = 1;
    oct::CalculateEdgeCount(edge_count.data(), inners.data(), num_brt_nodes);
  });

  // [Step 6.1] Prefix sum
  std::vector<int> oc_node_offsets(num_brt_nodes + 1);
  TimeTask("Prefix Sum", [&] {
    std::partial_sum(edge_count.begin(), edge_count.end(),
                     oc_node_offsets.begin() + 1);
    oc_node_offsets[0] = 0;
  });

  // [Step 6.2] Allocate BH nodes
  const int num_oc_nodes = oc_node_offsets.back();
  const int root_level = inners[0].delta_node / 3;
  Code_t root_prefix = morton_keys[0] >> (CODE_LEN - (3 * root_level));
  std::vector<oct::OctNode> bh_nodes(num_oc_nodes);

  // Debug print
  std::cout << "Num Morton Keys: " << morton_keys.size() << "\n";
  std::cout << "Num Radix Nodes: " << num_brt_nodes << "\n";
  std::cout << "Num Octree Nodes: " << num_oc_nodes << "\n";

  // [Step 7] Create unlinked BH nodes
  TimeTask("Make Unlinked BH nodes", [&] {
    oct::MakeNodes(bh_nodes.data(), oc_node_offsets.data(), edge_count.data(),
                   morton_keys.data(), inners.data(), num_brt_nodes, range);
  });

  // [Step 8] Linking BH nodes
  TimeTask("Link BH nodes", [&] {
    oct::LinkNodes(bh_nodes.data(), oc_node_offsets.data(), edge_count.data(),
                   morton_keys.data(), inners.data(), num_brt_nodes);
  });

  oct::CheckTree(root_prefix, root_level * 3, bh_nodes.data(), 0,
                 morton_keys.data());

  // for (int i = 0; i < num_oc_nodes; ++i) {
  //   std::cout << "OctNode " << i << "\n";
  //   std::cout << "\tchild_node_mask: "
  //             << std::bitset<8>(bh_nodes[i].child_node_mask) << "\n";
  //   std::cout << "\tchild_leaf_mask: "
  //             << std::bitset<8>(bh_nodes[i].child_leaf_mask) << "\n";

  //   std::cout << "\tchild : [" << bh_nodes[i].children[0];
  //   for (int j = 1; j < 8; ++j) {
  //     std::cout << ", " << bh_nodes[i].children[j];
  //   }
  //   std::cout << "]\n";

  //   std::cout << "\tcell_size: " << bh_nodes[i].cell_size << "\n";
  //   std::cout << "\tcornor: (" << bh_nodes[i].cornor.transpose() << ")\n";
  //   std::cout << "\n";
  // }

  return EXIT_SUCCESS;
}
