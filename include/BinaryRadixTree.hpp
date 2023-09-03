#pragma once

#include "Morton.hpp"

namespace brt {

struct InnerNodes {
  // The number of bits in the morton code, this node represents in [Karras]
  int delta_node;

  // pointers
  int left = -1;  // can be either inner or leaf
  int right = -1;
  int parent = -1;
};

}  // namespace brt

/**
 * @brief Given a sorted array of morton codes, make the nodes of binary radix.
 * The radix has 'n-1' internal nodes.
 *
 * @param sorted_morton sorted morton codes
 * @param num_unique_keys number of unique morton codes
 * @param brt_nodes output an array of internal nodes of size 'n-1'
 */
void BuildBinaryRadixTree(const Code_t* sorted_morton, size_t num_unique_keys,
                          brt::InnerNodes* brt_nodes);
