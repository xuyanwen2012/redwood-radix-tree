#pragma once

#include <Eigen/Dense>

#include "Utils.hpp"
#include "libmorton/morton.h"

using Code_t = uint64_t;
constexpr int kCodeLen = 63;

_NODISCARD inline Code_t PointToCode(const float x, const float y,
                                     const float z,
                                     const float min_coord = 0.0f,
                                     const float range = 1.0f) {
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  const auto x_coord =
      static_cast<uint32_t>(bit_scale * ((x - min_coord) / range));
  const auto y_coord =
      static_cast<uint32_t>(bit_scale * ((y - min_coord) / range));
  const auto z_coord =
      static_cast<uint32_t>(bit_scale * ((z - min_coord) / range));
  return libmorton::morton3D_64_encode(x_coord, y_coord, z_coord);
}

_NODISCARD inline Eigen::Vector3f CodeToPoint(const Code_t code,
                                              const float min_coord = 0.0f,
                                              const float range = 1.0f) {
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  uint_fast32_t dec_raw_x, dec_raw_y, dec_raw_z;
  libmorton::morton3D_64_decode(code, dec_raw_x, dec_raw_y, dec_raw_z);
  float dec_x = (static_cast<float>(dec_raw_x) / bit_scale) * range + min_coord;
  float dec_y = (static_cast<float>(dec_raw_y) / bit_scale) * range + min_coord;
  float dec_z = (static_cast<float>(dec_raw_z) / bit_scale) * range + min_coord;
  return {dec_x, dec_y, dec_z};
}
