#include <Eigen/Dense>
#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

#include "BinaryRadixTree.hpp"
#include "Common.hpp"
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
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(morton_keys),
                 [&](const auto& vec) {
                   const auto x = vec.x();
                   const auto y = vec.y();
                   const auto z = vec.z();
                   return MortonCode32(ToNBitInt(x), ToNBitInt(y),
                                       ToNBitInt(z));
                 });

  // [Step 2] Sort Morton Codes by Key
  std::sort(morton_keys.begin(), morton_keys.end());

  // [Step 3-4] Handle Duplicates
  morton_keys.erase(std::unique(morton_keys.begin(), morton_keys.end()),
                    morton_keys.end());

  std::for_each(morton_keys.begin(), morton_keys.end(), [](const auto key) {
    std::cout << key << "\t" << std::bitset<32>(key) << "\t" << std::endl;
  });

  // [Step 5] Build Binary Radix Tree
  constexpr auto num_brt_nodes = n - 1;
  std::vector<brt::InnerNodes> inners(num_brt_nodes);

  for (int i = 0; i < num_brt_nodes; ++i) {
    brt::MyProcessInternalNode(n, morton_keys.data(), i, inners.data());
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
  std::vector<int> oc_node_offsets(num_brt_nodes);  // aka node_offsets

  // Copy a "1" to the first element to account for the root
  edge_count[0] = 1;
  // root has no parent, so don't do for index 0
  for (int i = 1; i < num_brt_nodes; ++i) {
    const int my_depth = inners[i].delta_node / 3;
    const int parent_depth = inners[inners[i].parent].delta_node / 3;
    edge_count[i] = my_depth - parent_depth;
  }

  for (int i = 0; i < num_brt_nodes; ++i) {
    std::cout << "BrtNode " << i << " edge count: " << edge_count[i]
              << std::endl;
  }

  // prefix sum
  std::partial_sum(edge_count.begin(), edge_count.end(),
                   oc_node_offsets.begin());

  // [Step 6.1] Allocate BH nodes
  const int num_oc_nodes = oc_node_offsets.back() + 1;
  const auto root_delta = inners[0].delta_node;  // 1
  std::cout << "Num Octree Nodes: " << num_oc_nodes << "\n";
  std::cout << "Root delta: " << root_delta << "\n";

  // setup initial values of octree node objects
  std::vector<oct::OctNode> bh_nodes(num_oc_nodes);

  int root_level = inners[0].delta_node / 3;
  Code_t root_prefix = morton_keys[0] >> (CODE_LEN - (root_level * 3));
  std::cout << "root_level: " << root_level << "\n";
  std::cout << "root_prefix: " << root_prefix << " - "
            << std::bitset<32>(root_prefix) << "\n";

  // bh_nodes[0].body.pos = Eigen::Vector3f(0.5f, 0.5f, 0.5f);

  // skipping root
  for (int i = 1; i < num_oc_nodes; ++i) {
    int oct_idx = oc_node_offsets[i];
    int n_new_nodes = edge_count[i];
    for (int j = 0; j < n_new_nodes - 1; ++j) {
      int level = inners[i].delta_node / 3 - j;
      Code_t node_prefix = morton_keys[i] >> (CODE_LEN - (3 * level));
      int child_idx = node_prefix & 0b111;
      int parent = oct_idx + 1;
      bh_nodes[parent].setChild(oct_idx, child_idx);
      oct_idx = parent;
    }

    if (n_new_nodes > 0) {
      int rt_parent = inners[i].parent;
      while (edge_count[rt_parent] == 0) {
        rt_parent = inners[rt_parent].parent;
      }
      int oct_parent = oc_node_offsets[rt_parent];
      int top_level = inners[i].delta_node / 3 - n_new_nodes + 1;
      Code_t top_node_prefix = morton_keys[i] >> (CODE_LEN - (3 * top_level));
      int child_idx = top_node_prefix & 0b111;

      bh_nodes[oct_parent].setChild(oct_idx, child_idx);
    }
  }

  for (int i = 0; i < num_oc_nodes; ++i) {
    std::cout << "OctNode " << i << "\n";
    std::cout << "\tchild_node_mask: "
              << std::bitset<8>(bh_nodes[i].child_node_mask) << "\n";
    std::cout << "\n";
  }

  return EXIT_SUCCESS;
}
