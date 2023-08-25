#include "Octree.hpp"

#include <iostream>

#include "BinaryRadixTree.hpp"
namespace oct {

void OctNode::SetChild(const int child, const int my_child_idx) {
  children[my_child_idx] = child;
  // TODO: atomicOr in CUDA
  child_node_mask |= (1 << my_child_idx);
}

void OctNode::SetLeaf(const int leaf, const int my_child_idx) {
  children[my_child_idx] = leaf;
  // TODO: atomicOr in CUDA
  child_leaf_mask |= (1 << my_child_idx);
}

_NODISCARD bool IsLeaf(const int internal_value) {
  // check the most significant bit, which is used as a flag for "is leaf node"
  return internal_value >> (sizeof(int) * 8 - 1);
}

_NODISCARD int GetLeafIndex(const int internal_value) {
  // delete the last bit which tells if this is leaf or internal index
  return internal_value &
         ~(1 << (sizeof(int) * 8 -
                 1));  // NOLINT(clang-diagnostic-shift-sign-overflow)
}

void CalculateEdgeCount(int* edge_count, const brt::InnerNodes* inners,
                        const int num_brt_nodes) {
  // root has no parent, so don't do for index 0
  for (int i = 1; i < num_brt_nodes; ++i) {
    const int my_depth = inners[i].delta_node / 3;
    const int parent_depth = inners[inners[i].parent].delta_node / 3;
    edge_count[i] = my_depth - parent_depth;
  }
}

void MakeNodes(OctNode* nodes, const int* node_offsets, const int* edge_count,
               const Code_t* morton_keys, const brt::InnerNodes* inners,
               const int num_brt_nodes, const float tree_range) {
  // the root doesn't represent level 0 of the "entire" octree
  const int root_level = inners[0].delta_node / 3;
  const Code_t root_prefix = morton_keys[0] >> (CODE_LEN - (root_level * 3));

  nodes[0].cornor = CodeToPoint(root_prefix << (CODE_LEN - (root_level * 3)));
  nodes[0].cell_size = tree_range;

  // skipping root
  for (int i = 1; i < num_brt_nodes; ++i) {
    int oct_idx = node_offsets[i];
    const int n_new_nodes = edge_count[i];
    for (int j = 0; j < n_new_nodes - 1; ++j) {
      const int level = inners[i].delta_node / 3 - j;
      const Code_t node_prefix = morton_keys[i] >> (CODE_LEN - (3 * level));
      const int child_idx = static_cast<int>(node_prefix & 0b111);
      const int parent = oct_idx + 1;

      nodes[parent].SetChild(oct_idx, child_idx);

      // calculate corner point (LSB have already been shifted off)
      nodes[oct_idx].cornor =
          CodeToPoint(node_prefix << (CODE_LEN - (3 * level)));

      // each cell is half the size of the level above it
      nodes[oct_idx].cell_size =
          tree_range / static_cast<float>(1 << (level - root_level));

      oct_idx = parent;
    }

    if (n_new_nodes > 0) {
      int rt_parent = inners[i].parent;
      while (edge_count[rt_parent] == 0) {
        rt_parent = inners[rt_parent].parent;
      }
      const int oct_parent = node_offsets[rt_parent];
      const int top_level = inners[i].delta_node / 3 - n_new_nodes + 1;
      const Code_t top_node_prefix =
          morton_keys[i] >> (CODE_LEN - (3 * top_level));
      const int child_idx = static_cast<int>(top_node_prefix & 0b111);

      nodes[oct_parent].SetChild(oct_idx, child_idx);
      nodes[oct_idx].cornor =
          CodeToPoint(top_node_prefix << (CODE_LEN - (3 * top_level)));
      nodes[oct_idx].cell_size =
          tree_range / static_cast<float>(1 << (top_level - root_level));
    }
  }
}

void LinkNodes(OctNode* nodes, const int* node_offsets, const int* edge_count,
               const Code_t* morton_keys, const brt::InnerNodes* inners,
               const int num_brt_nodes) {
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
      const int bottom_oct_idx = node_offsets[rt_node];
      nodes[bottom_oct_idx].SetLeaf(leaf_idx, child_idx);
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
      const int bottom_oct_idx = node_offsets[rt_node];
      nodes[bottom_oct_idx].SetLeaf(leaf_idx, child_idx);
    }
  }
}

void CheckTree(const Code_t prefix, const int code_len, const OctNode* nodes,
               const int oct_idx, const Code_t* codes) {
  const OctNode& node = nodes[oct_idx];
  for (int i = 0; i < 8; ++i) {
    Code_t new_pref = (prefix << 3) | i;
    if (node.child_node_mask & (1 << i)) {
      CheckTree(new_pref, code_len + 3, nodes, node.children[i], codes);
    }
    if (node.child_leaf_mask & (1 << i)) {
      Code_t leaf_prefix =
          codes[node.children[i]] >> (CODE_LEN - (code_len + 3));
      if (new_pref != leaf_prefix) {
        printf("oh no...\n");
      }
    }
  }
}

}  // namespace oct