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

/**
 * @brief Count the number of Octree node under a binary radix tree node.
 *
 * @param edge_count: results edge count array
 * @param inners: binary radix tree nodes
 * @param num_brt_nodes: number of binary radix tree nodes
 */
void CalculateEdgeCount(int* edge_count, const brt::InnerNodes* inners,
                        int num_brt_nodes);

/**
 * @brief Make the unlinked octree nodes from the binary radix tree.
 * https://github.com/ahmidou/ShapeExtraction/blob/master/src/Octree.cu
 *
 * @param nodes: array of preallocated octree nodes
 * @param node_offsets: ranges of each RT node
 * @param edge_count: number of nodes in each RT node
 * @param morton_keys: sorted morton keys
 * @param inners: binary radix tree nodes
 * @param num_brt_nodes: number of binary radix tree nodes
 * @param min_cord: minimum coordinate of the entire octree, default 0.0f
 * @param tree_range: range of the entire octree, default 1.0f
 */
void MakeNodes(OctNode* nodes, const int* node_offsets, const int* edge_count,
               const Code_t* morton_keys, const brt::InnerNodes* inners,
               int num_brt_nodes, 
              //  float min_cord = 0.0f,
               float tree_range = 1.0f
               );

/**
 * @brief Link the octree nodes together.
 *
 * @param nodes: array of preallocated octree nodes
 * @param node_offsets: ranges of each RT node
 * @param edge_count: number of nodes in each RT node
 * @param morton_keys: sorted morton keys
 * @param inners: binary radix tree nodes
 * @param num_brt_nodes: number of binary radix tree nodes
 */
void LinkNodes(OctNode* nodes, const int* node_offsets, const int* edge_count,
               const Code_t* morton_keys, const brt::InnerNodes* inners,
               int num_brt_nodes);

void CheckTree(const Code_t prefix, int code_len, const OctNode* nodes,
               int oct_idx, const Code_t* codes);

}  // namespace oct