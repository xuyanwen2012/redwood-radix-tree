#include <iostream>

#include "BinaryRadixTree.hpp"
#include "Octree.hpp"
#include "cuda/CudaUtils.cuh"

__device__ void oct::OctNode::SetChild(const int child,
                                       const int my_child_idx) {
  children[my_child_idx] = child;
  // child_node_mask |= (1 << my_child_idx);
  atomicOr(&child_node_mask, 1 << my_child_idx);
}

__device__ void oct::OctNode::SetLeaf(const int leaf, const int my_child_idx) {
  children[my_child_idx] = leaf;
  // child_leaf_mask |= (1 << my_child_idx);
  atomicOr(&child_leaf_mask, 1 << my_child_idx);
}

_NODISCARD __device__ bool IsLeaf(const int internal_value) {
  // check the most significant bit, which is used as a flag for "is leaf
  // node"
  return internal_value >> (sizeof(int) * 8 - 1);
}

_NODISCARD __device__ int GetLeafIndex(const int internal_value) {
  // delete the last bit which tells if this is leaf or internal index
  return internal_value &
         ~(1 << (sizeof(int) * 8 -
                 1));  // NOLINT(clang-diagnostic-shift-sign-overflow)
}

__device__ void MakeNodesHelper(const int i, oct::OctNode* nodes,
                                const int* node_offsets, const int* edge_count,
                                const Code_t* morton_keys,
                                const brt::InnerNodes* inners,
                                const float min_coord, const float tree_range,
                                const int root_level) {
  int oct_idx = node_offsets[i];
  const int n_new_nodes = edge_count[i];
  for (int j = 0; j < n_new_nodes - 1; ++j) {
    const int level = inners[i].delta_node / 3 - j;
    const Code_t node_prefix = morton_keys[i] >> (kCodeLen - (3 * level));
    const int child_idx = static_cast<int>(node_prefix & 0b111);
    const int parent = oct_idx + 1;

    nodes[parent].SetChild(oct_idx, child_idx);

    // calculate corner point (LSB have already been shifted off)
    float dec_x, dec_y, dec_z;
    CodeToPoint(node_prefix << (kCodeLen - (3 * level)), dec_x, dec_y, dec_z,
                min_coord, tree_range);
    nodes[oct_idx].cornor = {dec_x, dec_y, dec_z};

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
        morton_keys[i] >> (kCodeLen - (3 * top_level));
    const int child_idx = static_cast<int>(top_node_prefix & 0b111);

    nodes[oct_parent].SetChild(oct_idx, child_idx);

    float dec_x, dec_y, dec_z;
    CodeToPoint(top_node_prefix << (kCodeLen - (3 * top_level)), dec_x, dec_y,
                dec_z, min_coord, tree_range);
    nodes[oct_idx].cornor = {dec_x, dec_y, dec_z};

    nodes[oct_idx].cell_size =
        tree_range / static_cast<float>(1 << (top_level - root_level));
  }
}

__global__ void MakeNodesKernel(oct::OctNode* nodes, const int* node_offsets,
                                const int* edge_count,
                                const Code_t* morton_keys,
                                const brt::InnerNodes* inners,
                                const size_t num_brt_nodes,
                                const float min_coord, const float tree_range,
                                const int root_level) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 0 && i < num_brt_nodes) {
    MakeNodesHelper(i, nodes, node_offsets, edge_count, morton_keys, inners,
                    min_coord, tree_range, root_level);
  }
}

void MakeUnlinkedOctreeNodes(oct::OctNode* nodes, const int* node_offset,
                             const int* edge_count, const Code_t* sorted_morton,
                             const brt::InnerNodes* brt_nodes,
                             const size_t num_brt_nodes, const float min_coord,
                             const float tree_range) {
  const auto root_level = brt_nodes[0].delta_node / 3;
  const Code_t root_prefix = sorted_morton[0] >> (kCodeLen - (root_level * 3));

  float dec_x, dec_y, dec_z;
  CodeToPoint(root_prefix << (kCodeLen - (root_level * 3)), dec_x, dec_y, dec_z,
              min_coord, tree_range);
  nodes[0].cornor = {dec_x, dec_y, dec_z};
  nodes[0].cell_size = tree_range;

  const auto num_blocks =
      (num_brt_nodes + kThreadsPerBlock - 1) / kThreadsPerBlock;  // round up

  MakeNodesKernel<<<num_blocks, kThreadsPerBlock>>>(
      nodes, node_offset, edge_count, sorted_morton, brt_nodes, num_brt_nodes,
      min_coord, tree_range, root_level);

  HANDLE_ERROR(cudaDeviceSynchronize());
}

// =================================================================================
//                                    Link Octree
// =================================================================================

__device__ void LinkNodesHelper(const int i, oct::OctNode* nodes,
                                const int* node_offsets, const int* edge_count,
                                const Code_t* morton_keys,
                                const brt::InnerNodes* inners) {
  if (IsLeaf(inners[i].left)) {
    const int leaf_idx = GetLeafIndex(inners[i].left);
    const int leaf_level = inners[i].delta_node / 3 + 1;
    const Code_t leaf_prefix =
        morton_keys[leaf_idx] >> (kCodeLen - (3 * leaf_level));

    const int child_idx = static_cast<int>(leaf_prefix & 0b111);
    // walk up the radix tree until finding a node which contributes
    // an octnode
    int rt_node = i;
    while (edge_count[rt_node] == 0) {
      rt_node = inners[rt_node].parent;
    }
    // the lowest octnode in the string contributed by rt_node will
    // be the lowest index
    const int bottom_oct_idx = node_offsets[rt_node];
    nodes[bottom_oct_idx].SetLeaf(leaf_idx, child_idx);
  }

  if (IsLeaf(inners[i].right)) {
    const int leaf_idx = GetLeafIndex(inners[i].left) + 1;
    const int leaf_level = inners[i].delta_node / 3 + 1;
    const Code_t leaf_prefix =
        morton_keys[leaf_idx] >> (kCodeLen - (3 * leaf_level));

    const int child_idx = static_cast<int>(leaf_prefix & 0b111);

    // walk up the radix tree until finding a node which contributes
    // an octnode
    int rt_node = i;
    while (edge_count[rt_node] == 0) {
      rt_node = inners[rt_node].parent;
    }

    // the lowest octnode in the string contributed by rt_node will
    // be the lowest index
    const int bottom_oct_idx = node_offsets[rt_node];
    nodes[bottom_oct_idx].SetLeaf(leaf_idx, child_idx);
  }
}

__global__ void LinkNodesKernel(oct::OctNode* nodes, const int* node_offsets,
                                const int* edge_count,
                                const Code_t* morton_keys,
                                const brt::InnerNodes* inners,
                                const size_t num_brt_nodes) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_brt_nodes) {
    LinkNodesHelper(i, nodes, node_offsets, edge_count, morton_keys, inners);
  }
}

void LinkOctreeNodes(oct::OctNode* nodes, const int* node_offsets,
                     const int* edge_count, const Code_t* sorted_morton,
                     const brt::InnerNodes* brt_nodes,
                     const size_t num_brt_nodes) {
  const auto num_blocks =
      (num_brt_nodes + kThreadsPerBlock - 1) / kThreadsPerBlock;  // round up
  LinkNodesKernel<<<num_blocks, kThreadsPerBlock>>>(
      nodes, node_offsets, edge_count, sorted_morton, brt_nodes, num_brt_nodes);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void CheckTree(const Code_t prefix, const int code_len,
               const oct::OctNode* nodes, const int oct_idx,
               const Code_t* codes) {
  const auto& node = nodes[oct_idx];
  for (int i = 0; i < 8; ++i) {
    Code_t new_pref = (prefix << 3) | i;
    if (node.child_node_mask & (1 << i)) {
      CheckTree(new_pref, code_len + 3, nodes, node.children[i], codes);
    }
    if (node.child_leaf_mask & (1 << i)) {
      Code_t leaf_prefix =
          codes[node.children[i]] >> (kCodeLen - (code_len + 3));
      if (new_pref != leaf_prefix) {
        printf("oh no...\n");
      }
    }
  }
}