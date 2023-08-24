#pragma once

#include <Eigen/Dense>

#include "BinaryRadixTree.hpp"

namespace oct {

struct Body {
  Eigen::Vector3f pos;
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
  void setChild(const int child, const int my_child_idx);

  /**
   * @brief Set the Leaf object
   *
   * @param leaf: index of point that will become the leaf child
   * @param my_child_idx: which of my children it will be [0-7]
   */
  void setLeaf(const int leaf, const int my_child_idx);
};

// OctNode MakeNode(const int root_delta, const Code_t root_prefix);

}  // namespace oct