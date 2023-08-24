#include <Eigen/Dense>
#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

#include "BinaryRadixTree.hpp"
#include "Morton.hpp"
#include "Octree.hpp"

void PrintVector3F(const Eigen::Vector3f& vec) {
  std::cout << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")\n";
}

int main() {
  thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution dis(0.0f, 1.0f);

  // Prepare Inputs
  constexpr int n = 128;
  std::vector<Eigen::Vector3f> inputs(n);
  std::generate(inputs.begin(), inputs.end(),
                [&] { return Eigen::Vector3f(dis(gen), dis(gen), dis(gen)); });

  // std::for_each(inputs.begin(), inputs.end(), PrintVector3F);

  // [Step 1] Compute Morton Codes
  std::vector<Code_t> morton_keys;
  morton_keys.reserve(n);
  std::transform(
      inputs.begin(), inputs.end(), std::back_inserter(morton_keys),
      [&](const auto& vec) { return PointToCode(vec.x(), vec.y(), vec.z()); });

  // [Step 2] Sort Morton Codes by Key
  std::sort(morton_keys.begin(), morton_keys.end());

  // [Step 3-4] Handle Duplicates
  morton_keys.erase(std::unique(morton_keys.begin(), morton_keys.end()),
                    morton_keys.end());

  std::for_each(morton_keys.begin(), morton_keys.end(), [](const auto key) {
    std::cout << key << "\t" << std::bitset<CODE_LEN>(key) << "\t"
              << CodeToPoint(key).transpose() << std::endl;
  });

  // [Step 5] Build Binary Radix Tree
  constexpr auto num_brt_nodes = n - 1;
  std::vector<brt::InnerNodes> inners(num_brt_nodes);

  for (int i = 0; i < num_brt_nodes; ++i) {
    MyProcessInternalNode(n, morton_keys.data(), i, inners.data());
  }

  for (int i = 0; i < num_brt_nodes; ++i) {
    std::cout << "Node " << i << "\n";
    std::cout << "\tdelta_node: " << inners[i].delta_node << "\n";
    std::cout << "\tsfc_code: " << inners[i].sfc_code << "\n";
    std::cout << "\tleft: " << inners[i].left << "\n";
    std::cout << "\tright: " << inners[i].right << "\n";
    std::cout << "\tparent: " << inners[i].parent << "\n";
    std::cout << "\n";
  }

  // [Step 6] Count edges
  std::vector<int> edge_count(num_brt_nodes);
  oct::CalculateEdgeCount(edge_count.data(), inners.data(), num_brt_nodes);

  // [Step 6.1] Prefix sum
  std::vector<int> oc_node_offsets(num_brt_nodes);
  std::partial_sum(edge_count.begin(), edge_count.end(),
                   oc_node_offsets.begin());

  // for (int i = 0; i < num_brt_nodes; ++i) {
  //   std::cout << "[debug]\t" << oc_node_offsets[i] << std::endl;
  // }

  // [Step 6.2] Allocate BH nodes
  const int num_oc_nodes = oc_node_offsets.back() + 1;
  const int root_level = inners[0].delta_node / 3;
  Code_t root_prefix = morton_keys[0] >> (CODE_LEN - (3 * root_level));
  std::vector<oct::OctNode> bh_nodes(num_oc_nodes);

  // Debug print
  std::cout << "Num Morton Keys: " << morton_keys.size() << "\n";
  std::cout << "Num Radix Nodes: " << num_brt_nodes << "\n";
  std::cout << "Num Octree Nodes: " << num_oc_nodes << "\n";

  // [Step 7] Create unlinked BH nodes
  oct::MakeNodes(bh_nodes.data(), oc_node_offsets.data(), edge_count.data(),
                 morton_keys.data(), inners.data(), num_brt_nodes);

  // [Step 8] Linking BH nodes
  oct::LinkNodes(bh_nodes.data(), oc_node_offsets.data(), edge_count.data(),
                 morton_keys.data(), inners.data(), num_brt_nodes);

  oct::CheckTree(root_prefix, root_level * 3, bh_nodes.data(), 0,
                 morton_keys.data());

  for (int i = 0; i < num_oc_nodes; ++i) {
    std::cout << "OctNode " << i << "\n";
    std::cout << "\tchild_node_mask: "
              << std::bitset<8>(bh_nodes[i].child_node_mask) << "\n";
    std::cout << "\tchild_leaf_mask: "
              << std::bitset<8>(bh_nodes[i].child_leaf_mask) << "\n";

    std::cout << "\tchild : [" << bh_nodes[i].children[0];
    for (int j = 1; j < 8; ++j) {
      std::cout << ", " << bh_nodes[i].children[j];
    }
    std::cout << "]\n";

    std::cout << "\tcell_size: " << bh_nodes[i].cell_size << "\n";
    std::cout << "\tcornor: (" << bh_nodes[i].cornor.transpose() << ")\n";
    std::cout << "\n";
  }

  return EXIT_SUCCESS;
}
