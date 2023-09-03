#pragma once

#include "BinaryRadixTree.hpp"

/**
 * @brief Count the number of octree nodes under each binary radix tree node.
 *
 * @param brt_nodes
 * @param edge_count
 * @param num_nodes
 */
void CalculateEdgeCount(const brt::InnerNodes* brt_nodes, int* edge_count,
                        size_t num_nodes);

/**
 * @brief Compute the offset of each binary radix tree node.
 *
 * @param edge_count
 * @param node_offset
 * @param num_nodes
 */
void ComputeRangeArray(const int* edge_count, int* node_offset,
                       size_t num_nodes);
