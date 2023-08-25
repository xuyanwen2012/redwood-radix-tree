#pragma once

#include <Eigen/Dense>

#include "Utils.hpp"
#include "libmorton/morton.h"

using Code_t = uint64_t;
constexpr int CODE_LEN = 63;

_NODISCARD inline Code_t PointToCode(const float x, const float y,
                                     const float z,
                                     const float min_coord = 0.0f,
                                     const float range = 1.0f) {
  constexpr uint32_t bitscale = 0xFFFFFFFFu >> (32 - (CODE_LEN / 3));
  const uint32_t x_coord =
      static_cast<uint32_t>(bitscale * ((x - min_coord) / range));
  const uint32_t y_coord =
      static_cast<uint32_t>(bitscale * ((y - min_coord) / range));
  const uint32_t z_coord =
      static_cast<uint32_t>(bitscale * ((z - min_coord) / range));
  return libmorton::morton3D_64_encode(x_coord, y_coord, z_coord);
}

_NODISCARD inline Eigen::Vector3f CodeToPoint(const Code_t code,
                                              const float min_coord = 0.0f,
                                              const float range = 1.0f) {
  constexpr uint32_t bitscale = 0xFFFFFFFFu >> (32 - (CODE_LEN / 3));
  uint_fast32_t dec_raw_x, dec_raw_y, dec_raw_z;
  libmorton::morton3D_64_decode(code, dec_raw_x, dec_raw_y, dec_raw_z);
  float dec_x = ((float)dec_raw_x / bitscale) * range + min_coord;
  float dec_y = ((float)dec_raw_y / bitscale) * range + min_coord;
  float dec_z = ((float)dec_raw_z / bitscale) * range + min_coord;
  return {dec_x, dec_y, dec_z};
}
