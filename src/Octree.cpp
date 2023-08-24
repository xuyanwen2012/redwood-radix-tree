#include "Octree.hpp"

namespace oct {

void OctNode::SetChild(const int child, const int my_child_idx) {
  children[my_child_idx] = child;
  // TODO: atomicOr in CUDA
  child_node_mask |= (1 << my_child_idx);
}

void OctNode::SetLeaf(const int leaf, const int my_child_idx) {
  children[my_child_idx] = leaf;
  // TODO: atomicOr in CUDA
  child_leaf_mask &= ~(1 << my_child_idx);
}

}  // namespace oct