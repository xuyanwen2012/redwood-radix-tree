#pragma once

#include "Morton.hpp"
#include "Utils.hpp"

namespace brt {

struct InnerNodes {
  // The number of bits in the morton code, this node represents in [Karras]
  int delta_node;

  // pointers
  int left = -1;  // can be either inner or leaf
  int right = -1;
  int parent = -1;
};

/**
 * @brief Given a sorted array of morton codes, make the nodes of binary radix
 * tree. The radix has 'n-1' internal nodes.
 *
 * @param key_num: number of sorted morton codes
 * @param morton_keys: sorted morton codes
 * @param brt_nodes: output an array of internal nodes of size 'n-1'
 */
void ProcessInternalNodes(int key_num, const Code_t* morton_keys,
                          InnerNodes* brt_nodes);

}  // namespace brt
