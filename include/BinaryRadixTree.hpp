#pragma once

#include <bits/stdint-uintn.h>

#include "Common.hpp"
#include "Morton.hpp"

namespace math {
template <typename T>
int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
T min(const T& x, const T& y) {
  return y ^ ((x ^ y) & -(x < y));
}

template <typename T>
T max(const T& x, const T& y) {
  return x ^ ((x ^ y) & -(x < y));
}

template <typename T>
int divideceil(const T& x, const T& y) {
  return (x + y - 1) / y;
}

/** Integer division by two, rounding up */
template <typename T>
int divide2ceil(const T& x) {
  return (x + 1) >> 1;
}
}  // namespace math

namespace brt {

// This is the easier version
struct InnerNodes {
  // 31-bit morton code, packed to the right (or 63-bit)
  Code_t sfc_code;

  // The number of bits in the morton code, this node represents in [Karras]
  int delta_node;

  // pointers
  int left;  // can be either inner or leaf
  int right;
  int parent = -1;
};

_NODISCARD inline int Delta(const Code_t* morton_keys, const int i,
                            const int j) noexcept {
  const auto li = morton_keys[i];
  const auto lj = morton_keys[j];
  return CommonPrefix(li, lj);
}

_NODISCARD inline int DeltaSafe(const int key_num, const Code_t* morton_keys,
                                const int i, const int j) noexcept {
  return (j < 0 || j >= key_num) ? -1 : Delta(morton_keys, i, j);
}

void MyProcessInternalNode(int key_num, const Code_t* morton_keys, const int i,
                           InnerNodes* brt_nodes);

namespace node {
template <typename T>
// index to a leaf node. difference between a leaf node index and an internal
// node index is that the leaf's node index most significant bit is 1
_NODISCARD inline T make_leaf(const T& index) {
  return index ^ (-1 ^ index) & (1UL << ((sizeof(T) * 8 - 1)));
}

template <typename T>  // index to an internal node
_NODISCARD inline T make_internal(const T& index) {
  return index;
}
}  // namespace node

}  // namespace brt