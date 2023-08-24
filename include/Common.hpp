#pragma once

#include <cmath>
#ifdef _MSC_VER
#include <immintrin.h>
#endif

#include "Utils.hpp"

// for 32-bit morton code
constexpr auto kMaxTreeLevel = 10;
constexpr auto kUnusedBits = 1;

_NODISCARD inline unsigned ToNBitInt(const float x) {
  constexpr auto n_bits = kMaxTreeLevel;

  const auto result = static_cast<unsigned>(x * (1u << n_bits));
  return std::min(result, (1u << n_bits) - 1u);
}

_NODISCARD inline unsigned CountLeadingZeros(const uint32_t num) {
#ifdef __CUDA_ARCH__
  return __clz(num);
#elif __GNUC__ || __clang__
  return __builtin_clz(num);
#elif _MSC_VER
  return _lzcnt_u32(num);
#endif
}

_NODISCARD inline unsigned CommonPrefix(const uint32_t i, const uint32_t j) {
  return CountLeadingZeros(i ^ j) - kUnusedBits;
}
