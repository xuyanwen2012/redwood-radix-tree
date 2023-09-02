#include <cassert>
#include <cub/cub.cuh>

#include "PrefixSum.hpp"
#include "cuda/CudaUtils.cuh"

void PrefixSum(const int* input, int* output, const int n) {
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, input,
                                output, n);

  HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, input,
                                output, n);
  HANDLE_ERROR(cudaDeviceSynchronize());
  // HANDLE_ERROR(cudaFree(d_temp_storage));
}

void ComputeRangeArray(const redwood::UsmVector<int>& edge_counts,
                       redwood::UsmVector<int>& oc_node_offset) {
  assert(oc_node_offset.size() == edge_counts.size() + 1);
  PrefixSum(edge_counts.data(), oc_node_offset.data() + 1, edge_counts.size());
  // To turn this prefix sum array into a range array, we need to shift it
  oc_node_offset[0] = 0;
}