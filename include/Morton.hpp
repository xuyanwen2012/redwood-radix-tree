#pragma once

#include "Utils.hpp"

using Code_t = uint32_t;
constexpr int CODE_LEN = 31;

namespace detail {
// Expand 10-bit int to 30-bits
_NODISCARD constexpr uint32_t ExpandBits32(uint32_t v) {
  v &= 0x000003ffu;  // discard bit higher than 10
  v = v * 0x00010001u & 0xFF0000FFu;
  v = v * 0x00000101u & 0x0F00F00Fu;
  v = v * 0x00000011u & 0xC30C30C3u;
  v = v * 0x00000005u & 0x49249249u;
  return v;
}
}  // namespace detail

// Make sure each 'xyz' value is in range [0, 1023]
_NODISCARD constexpr Code_t MortonCode32(const uint32_t ix, const uint32_t iy,
                                         const uint32_t iz) noexcept {
  const auto xx = detail::ExpandBits32(ix);
  const auto yy = detail::ExpandBits32(iy);
  const auto zz = detail::ExpandBits32(iz);
  return xx * 4 + yy * 2 + zz;
}
