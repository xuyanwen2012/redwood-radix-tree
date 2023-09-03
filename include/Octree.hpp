#pragma once

#include <Eigen/Dense>

#include "BinaryRadixTree.hpp"

namespace oct {

struct Body {
  float mass;
};

struct OctNode {
  // Payload
  Body body;

  Eigen::Vector3f cornor;
  float cell_size;

  // TODO: This is overkill number of pointers
  int children[8];

  /**
   * @brief For bit position i (from the right): If 1, children[i] is the index
   * of a child octree node. If 0, the ith child is either absent, or
   * children[i] is the index of a leaf.
   */
  int child_node_mask;

  /**
   * @brief For bit position i (from the right): If 1, children[i] is the index
   * of a leaf (in the corresponding points array). If 0, the ith child is
   * either absent, or an octree node.
   */
  int child_leaf_mask;

  /**
   * @brief Set a child
   *
   * @param child: index of octree node that will become the child
   * @param my_child_idx: which of my children it will be [0-7]
   */
  void SetChild(const int child, const int my_child_idx);

  /**
   * @brief Set the Leaf object
   *
   * @param leaf: index of point that will become the leaf child
   * @param my_child_idx: which of my children it will be [0-7]
   */
  void SetLeaf(const int leaf, const int my_child_idx);
};

}  // namespace oct

/**
 * @brief Make the unlinked octree nodes from the binary radix tree.
 *
 * @param nodes
 * @param node_offset
 * @param edge_count
 * @param sorted_morton
 * @param brt_nodes
 * @param num_brt_nodes
 * @param min_coord
 * @param tree_range
 */
void MakeUnlinkedOctreeNodes(oct::OctNode* nodes, const int* node_offset,
                             const int* edge_count, const Code_t* sorted_morton,
                             const brt::InnerNodes* brt_nodes,
                             size_t num_brt_nodes, float min_coord = 0.0f,
                             float tree_range = 1.0f);

/**
 * @brief Link the octree nodes together.
 *
 * @param nodes
 * @param node_offsets
 * @param edge_count
 * @param sorted_morton
 * @param brt_nodes
 * @param num_brt_nodes
 */
void LinkOctreeNodes(oct::OctNode* nodes, const int* node_offsets,
                     const int* edge_count, const Code_t* sorted_morton,
                     const brt::InnerNodes* brt_nodes, size_t num_brt_nodes);
/**
 * @brief Check that the octree is correct.
 *
 * @param prefix
 * @param code_len
 * @param nodes
 * @param oct_idx
 * @param codes
 */
void CheckTree(Code_t prefix, int code_len, const oct::OctNode* nodes,
               int oct_idx, const Code_t* codes);
