#pragma once

#include <algorithm>

#ifdef _MSC_VER
#include <immintrin.h>
#endif

#include "Utils.hpp"

/**
 * @brief Count the number of leading zeros in a 32-bit number.
 *
 * @param num input number
 * @return number of leading zeros
 */
_NODISCARD inline unsigned CountLeadingZeros(const uint32_t num) {
#ifdef __CUDA_ARCH__
  return __clz(num);
#elif __GNUC__ || __clang__
  return __builtin_clz(num);
#elif _MSC_VER
  return _lzcnt_u32(num);
#endif
}

/**
 * @brief Calculate the number of common prefix bits between two number.
 *
 * @param i first number
 * @param j second number
 * @return number of common prefix bits
 */
_NODISCARD inline unsigned CommonPrefix(const uint32_t i, const uint32_t j) {
  constexpr auto unused_bits = 1;
  return CountLeadingZeros(i ^ j) - unused_bits;
}
