#pragma once

// #include "Common.hpp"
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

/**
 * @brief Calculate the number of common prefix bits between two morton codes.
 *
 * @param morton_keys: sorted (not necessary) morton codes
 * @param i: index of the first morton code
 * @param j: index of the second morton code
 * @return number of common prefix bits
 */
_NODISCARD __device__ int Delta(const Code_t* morton_keys, const int i,
                                const int j);

// _NODISCARD  __device__ inline int Delta(const Code_t* morton_keys,
//                                                 const int i,
//                                                 const int j) noexcept {
//   const auto li = morton_keys[i];
//   const auto lj = morton_keys[j];
//   return CommonPrefix(li, lj);
// }
/**
 * @brief Calculate the number of common prefix bits between two morton codes.
 * Safe version, return -1 if the index is out of range.
 *
 * @param key_num: Number of morton codes (after remove duplicate)
 * @param morton_keys: sorted (not necessary) morton codes
 * @param i: index of the first morton code
 * @param j: index of the second morton code
 * @return number of common prefix bits
 */
_NODISCARD __device__ inline int DeltaSafe(const int key_num,
                                           const Code_t* morton_keys,
                                           const int i, const int j) noexcept {
  return (j < 0 || j >= key_num) ? -1 : Delta(morton_keys, i, j);
}

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

namespace node {

template <typename T>
_NODISCARD __device__ T make_leaf(const T& index) {
  return index ^ ((-1 ^ index) & 1UL << (sizeof(T) * 8 - 1));
}

template <typename T>
_NODISCARD __device__ T make_internal(const T& index) {
  return index;
}
}  // namespace node

}  // namespace brt

namespace math {
template <typename T>
__device__ int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
__device__ T min(const T& x, const T& y) {
  return y ^ ((x ^ y) & -(x < y));
}

template <typename T>
__device__ T max(const T& x, const T& y) {
  return x ^ ((x ^ y) & -(x < y));
}

template <typename T>
__device__ int divide_ceil(const T& x, const T& y) {
  return (x + y - 1) / y;
}

/** Integer division by two, rounding up */
template <typename T>
__device__ int divide2_ceil(const T& x) {
  return (x + 1) >> 1;
}
}  // namespace math
