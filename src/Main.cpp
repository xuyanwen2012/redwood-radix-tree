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

_NODISCARD bool IsLeaf(const int internal_value) {
  // check the most significant bit, which is used as a flag for "is leaf node"
  return internal_value >> (sizeof(int) * 8 - 1);
}

_NODISCARD int GetLeafIndex(const int internal_value) {
  // delete the last bit which tells if this is leaf or internal index
  return internal_value & ~(1 << (sizeof(int) * 8 - 1));   // NOLINT(clang-diagnostic-shift-sign-overflow)
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

  // Debug print
  std::cout << "Num Morton Keys: " << morton_keys.size() << "\n";
  std::cout << "Num Radix Nodes: " << num_brt_nodes << "\n";
  std::cout << "Num Octree Nodes: " << num_oc_nodes << "\n";
  std::cout << "Root delta: " << root_delta << "\n";

  // setup initial values of octree node objects
  std::vector<oct::OctNode> bh_nodes(num_oc_nodes);

  constexpr auto tree_range = 1.0f;
  const int root_level = inners[0].delta_node / 3;
  const Code_t root_prefix = morton_keys[0] >> (CODE_LEN - (root_level * 3));
  std::cout << "root_level: " << root_level << "\n";
  std::cout << "root_prefix: " << root_prefix << " - "
            << std::bitset<CODE_LEN>(root_prefix) << "\n";

  bh_nodes[0].cornor =
      CodeToPoint(root_prefix << (CODE_LEN - (3 * root_level)));
  bh_nodes[0].cell_size = tree_range;

  // skipping root
  // [Step 7] Creating unlinked BH nodes
  // https://github.com/ahmidou/ShapeExtraction/blob/master/src/Octree.cu
  for (int i = 1; i < num_brt_nodes; ++i) {
    int oct_idx = oc_node_offsets[i];
    const int n_new_nodes = edge_count[i];
    for (int j = 0; j < n_new_nodes - 1; ++j) {
      const int level = inners[i].delta_node / 3 - j;
      const Code_t node_prefix = morton_keys[i] >> (CODE_LEN - (3 * level));
      const int child_idx = static_cast<int>(node_prefix & 0b111);
      const int parent = oct_idx + 1;
      bh_nodes[parent].SetChild(oct_idx, child_idx);

      bh_nodes[oct_idx].cornor =
          CodeToPoint(node_prefix << (CODE_LEN - (3 * level)));
      bh_nodes[oct_idx].cell_size =
          tree_range / static_cast<float>(1 << (level - root_level));

      oct_idx = parent;
    }

    if (n_new_nodes > 0) {
      int rt_parent = inners[i].parent;
      while (edge_count[rt_parent] == 0) {
        rt_parent = inners[rt_parent].parent;
      }
      const int oct_parent = oc_node_offsets[rt_parent];
      const int top_level = inners[i].delta_node / 3 - n_new_nodes + 1;
      const Code_t top_node_prefix =
          morton_keys[i] >> (CODE_LEN - (3 * top_level));
      const int child_idx = static_cast<int>(top_node_prefix & 0b111);

      bh_nodes[oct_parent].SetChild(oct_idx, child_idx);

      bh_nodes[oct_idx].cornor =
          CodeToPoint(top_node_prefix << (CODE_LEN - (3 * top_level)));
      bh_nodes[oct_idx].cell_size =
          tree_range / static_cast<float>(1 << (top_level - root_level));
    }
  }

  // [Step 7] Linking BH nodes
  for (int i = 0; i < num_brt_nodes; ++i) {
    if (IsLeaf(inners[i].left)) {
      const int leaf_idx = GetLeafIndex(inners[i].left);
      const int leaf_level = inners[i].delta_node / 3 + 1;
      const Code_t leaf_prefix =
          morton_keys[leaf_idx] >> (CODE_LEN - (3 * leaf_level));

      const int child_idx = static_cast<int>(leaf_prefix & 0b111);
      // walk up the radix tree until finding a node which contributes an
      // octnode
      int rt_node = i;
      while (edge_count[rt_node] == 0) {
        rt_node = inners[rt_node].parent;
      }
      // the lowest octnode in the string contributed by rt_node will be the
      // lowest index
      const int bottom_oct_idx = oc_node_offsets[rt_node];
      bh_nodes[bottom_oct_idx].SetLeaf(leaf_idx, child_idx);
    }

    if (IsLeaf(inners[i].right)) {
      const int leaf_idx = GetLeafIndex(inners[i].left) + 1;
      const int leaf_level = inners[i].delta_node / 3 + 1;
      const Code_t leaf_prefix =
          morton_keys[leaf_idx] >> (CODE_LEN - (3 * leaf_level));

      const int child_idx = static_cast<int>(leaf_prefix & 0b111);

      // walk up the radix tree until finding a node which contributes an
      // octnode
      int rt_node = i;
      while (edge_count[rt_node] == 0) {
        rt_node = inners[rt_node].parent;
      }
      // the lowest octnode in the string contributed by rt_node will be the
      // lowest index
      const int bottom_oct_idx = oc_node_offsets[rt_node];
      bh_nodes[bottom_oct_idx].SetLeaf(leaf_idx, child_idx);
    }
  }

  for (int i = 0; i < num_oc_nodes; ++i) {
    std::cout << "OctNode " << i << "\n";
    std::cout << "\tchild_node_mask: "
              << std::bitset<8>(bh_nodes[i].child_node_mask) << "\n";
    std::cout << "\tcell_size: " << bh_nodes[i].cell_size << "\n";
    std::cout << "\tcornor: (" << bh_nodes[i].cornor.transpose() << ")\n";
    std::cout << "\n";
  }

  return EXIT_SUCCESS;
}
