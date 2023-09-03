#pragma once

#include <cstdint>

#include "Utils.hpp"

using Code_t = uint64_t;
constexpr int kCodeLen = 63;

_NODISCARD constexpr uint64_t ExpandBits64(const uint32_t a) noexcept {
  uint64_t x = static_cast<uint64_t>(a) & 0x1fffff;
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

_NODISCARD constexpr uint64_t Encode64(const uint32_t x, const uint32_t y,
                                       const uint32_t z) noexcept {
  return ExpandBits64(x) | (ExpandBits64(y) << 1) | (ExpandBits64(z) << 2);
}

_NODISCARD constexpr uint32_t CompressBits64(const uint64_t m) noexcept {
  uint64_t x = m & 0x1249249249249249;
  x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
  x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
  x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
  x = (x ^ (x >> 16)) & 0x1f00000000ffff;
  x = (x ^ (x >> 32)) & 0x1fffff;
  return static_cast<uint32_t>(x);
}

constexpr void Decode64(const uint64_t m, uint32_t& x, uint32_t& y,
                        uint32_t& z) noexcept {
  x = CompressBits64(m);
  y = CompressBits64(m >> 1);
  z = CompressBits64(m >> 2);
}

_NODISCARD constexpr Code_t PointToCode(const float x, const float y,
                                        const float z, const float min_coord,
                                        const float range) {
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  const auto x_coord =
      static_cast<uint32_t>(bit_scale * ((x - min_coord) / range));
  const auto y_coord =
      static_cast<uint32_t>(bit_scale * ((y - min_coord) / range));
  const auto z_coord =
      static_cast<uint32_t>(bit_scale * ((z - min_coord) / range));

  return Encode64(x_coord, y_coord, z_coord);
}

constexpr void CodeToPoint(const Code_t code, float& dec_x, float& dec_y,
                           float& dec_z, const float min_coord,
                           const float range) {
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  uint32_t dec_raw_x = 0, dec_raw_y = 0, dec_raw_z = 0;
  Decode64(code, dec_raw_x, dec_raw_y, dec_raw_z);
  dec_x = (static_cast<float>(dec_raw_x) / bit_scale) * range + min_coord;
  dec_y = (static_cast<float>(dec_raw_y) / bit_scale) * range + min_coord;
  dec_z = (static_cast<float>(dec_raw_z) / bit_scale) * range + min_coord;
}