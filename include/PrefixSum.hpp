#pragma once

#include "UnifiedSharedMemory.hpp"

void PrefixSum(const int* input, int* output, int n);

void ComputeRangeArray(const redwood::UsmVector<int>& edge_counts,
                       redwood::UsmVector<int>& oc_node_offset);
