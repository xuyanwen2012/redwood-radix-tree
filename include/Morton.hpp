#pragma once

#include <Eigen/Dense>

#include "Utils.hpp"

using Code_t = uint64_t;
constexpr int kCodeLen = 63;

_NODISCARD constexpr inline uint64_t ExpandBits64(const uint32_t a) {
  uint64_t x = static_cast<uint64_t>(a) & 0x1fffff;
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

_NODISCARD constexpr inline uint64_t Encode64(const uint32_t x,
                                              const uint32_t y,
                                              const uint32_t z) {
  return ExpandBits64(x) | (ExpandBits64(y) << 1) | (ExpandBits64(z) << 2);
}

_NODISCARD constexpr inline uint32_t CompressBits64(const uint64_t m) {
  uint64_t x = m & 0x1249249249249249;
  x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
  x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
  x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
  x = (x ^ (x >> 16)) & 0x1f00000000ffff;
  x = (x ^ (x >> 32)) & 0x1fffff;
  return static_cast<uint32_t>(x);
}

constexpr inline void Decode64(const uint64_t m, uint32_t& x, uint32_t& y,
                               uint32_t& z) {
  x = CompressBits64(m);
  y = CompressBits64(m >> 1);
  z = CompressBits64(m >> 2);
}

_NODISCARD _HOST_DEVICE Code_t PointToCode(float x, float y, float z,
                                           float min_coord = 0.0f,
                                           float range = 1.0f);

_NODISCARD _HOST_DEVICE Eigen::Vector3f CodeToPoint(Code_t code,
                                                    float min_coord = 0.0f,
                                                    float range = 1.0f);

void ComputeMortonCodes(const Eigen::Vector3f* inputs, Code_t* morton_keys,
                        int n, float min_coord = 0.0f, float range = 1.0f);

void SortMortonCodes(const Code_t* morton_keys, Code_t* sorted_morton_keys,
                     int n);