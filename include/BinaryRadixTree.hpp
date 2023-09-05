#pragma once

#include "Morton.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define CLZ(x) __builtin_clzll(x)
#elif defined(__CUDACC__)
#define CLZ(x) __clzll(x)
#elif defined(_MSC_VER)
#include <intrin.h>
#define CLZ(x) __lzcnt64(x);
#else
#error "Unsupported compiler"
#endif

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#define DEVICE __device__
#define DEVICE_INLINE __device__ __forceinline__
#else
#define HOST_DEVICE
#define DEVICE
#define DEVICE_INLINE
#endif

namespace brt {

struct InnerNodes {
  // The number of bits in the morton code, this node represents in [Karras]
  int delta_node;

  // pointers
  int left = -1;  // can be either inner or leaf
  int right = -1;
  int parent = -1;
};

namespace math {
template <typename T>
_NODISCARD constexpr int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
_NODISCARD constexpr T min(const T& x, const T& y) {
  return y ^ ((x ^ y) & -(x < y));
}

template <typename T>
_NODISCARD constexpr T max(const T& x, const T& y) {
  return x ^ ((x ^ y) & -(x < y));
}

template <typename T>
_NODISCARD constexpr int divide_ceil(const T& x, const T& y) {
  return (x + y - 1) / y;
}

/** Integer division by two, rounding up */
template <typename T>
_NODISCARD constexpr int divide2_ceil(const T& x) {
  return (x + 1) >> 1;
}
}  // namespace math

namespace node {
template <typename T>
_NODISCARD constexpr T make_leaf(const T& index) {
  return index ^ ((-1 ^ index) & 1UL << (sizeof(T) * 8 - 1));
}

template <typename T>
_NODISCARD constexpr T make_internal(const T& index) {
  return index;
}
}  // namespace node

_NODISCARD constexpr int Delta(const Code_t* morton_keys, const int i,
                               const int j) {
  constexpr auto unused_bits = 1;
  const auto li = morton_keys[i];
  const auto lj = morton_keys[j];
  return CLZ(li ^ lj) - unused_bits;
}

_NODISCARD constexpr int DeltaSafe(const int key_num, const Code_t* morton_keys,
                                   const int i, const int j) {
  return (j < 0 || j >= key_num) ? -1 : Delta(morton_keys, i, j);
}

DEVICE inline void ProcessInternalNodesHelper(const int key_num,
                                              const Code_t* morton_keys,
                                              const int i,
                                              InnerNodes* brt_nodes) {
  const auto direction{
      math::sign<int>(Delta(morton_keys, i, i + 1) -
                      DeltaSafe(key_num, morton_keys, i, i - 1))};
  // assert(direction != 0);

  const auto delta_min{DeltaSafe(key_num, morton_keys, i, i - direction)};

  int I_max{2};
  while (DeltaSafe(key_num, morton_keys, i, i + I_max * direction) > delta_min)
    I_max <<= 2;  // aka, *= 2

  // Find the other end using binary search.
  int I{0};
  for (int t{I_max / 2}; t; t /= 2)
    if (DeltaSafe(key_num, morton_keys, i, i + (I + t) * direction) > delta_min)
      I += t;

  const int j{i + I * direction};

  // Find the split position using binary search.
  const auto delta_node{DeltaSafe(key_num, morton_keys, i, j)};
  auto s{0};

  int t{I};
  do {
    t = math::divide2_ceil<int>(t);
    if (DeltaSafe(key_num, morton_keys, i, i + (s + t) * direction) >
        delta_node)
      s += t;
  } while (t > 1);

  const auto split{i + s * direction + math::min<int>(direction, 0)};

  // // sanity check
  // assert(Delta(morton_keys, i, j) > delta_min);
  // assert(Delta(morton_keys, split, split + 1) ==
  // Delta(morton_keys, i, j)); assert(!(split < 0 || split + 1 >=
  // key_num));

  const int left{math::min<int>(i, j) == split
                     ? node::make_leaf<int>(split)
                     : node::make_internal<int>(split)};

  const int right{math::max<int>(i, j) == split + 1
                      ? node::make_leaf<int>(split + 1)
                      : node::make_internal<int>(split + 1)};

  brt_nodes[i].delta_node = delta_node;
  brt_nodes[i].left = left;
  brt_nodes[i].right = right;

  if (math::min<int>(i, j) != split) brt_nodes[left].parent = i;
  if (math::max<int>(i, j) != split + 1) brt_nodes[right].parent = i;
}

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
