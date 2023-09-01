#pragma once

#include <Eigen/Dense>

#include "Utils.hpp"
#include "cuda/CudaUtils.cuh"
#include "libmorton/morton.h"

using Code_t = uint64_t;
constexpr int kCodeLen = 63;

__device__ __host__ inline uint64_t expandBits64(uint32_t v) {
  v = (v * 0x000100000001u) & 0xFFFF00000000FFFFu;
  v = (v * 0x000000010001u) & 0x00FF0000FF0000FFu;
  v = (v * 0x000000000101u) & 0xF00F00F00F00F00Fu;
  v = (v * 0x000000000011u) & 0x30C30C30C30C30C3u;
  v = (v * 0x000000000005u) & 0x9249249249249249u;

  return v;
}

__device__ __host__ inline uint64_t Morton(const uint32_t x, const uint32_t y,
                                           const uint32_t z) {
  uint64_t xx = expandBits64(x);
  uint64_t yy = expandBits64(y);
  uint64_t zz = expandBits64(z);
  return xx * 4 + yy * 2 + zz;
}

__device__ __host__ inline Code_t MyVersionPointToCode(
    const float x, const float y, const float z, const float min_coord = 0.0f,
    const float range = 1.0f) {
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  const auto x_coord =
      static_cast<uint32_t>(bit_scale * ((x - min_coord) / range));
  const auto y_coord =
      static_cast<uint32_t>(bit_scale * ((y - min_coord) / range));
  const auto z_coord =
      static_cast<uint32_t>(bit_scale * ((z - min_coord) / range));
  return Morton(x_coord, y_coord, z_coord);
}

__device__ __host__ inline Code_t PointToCode(const float x, const float y,
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

__device__ __host__ inline Eigen::Vector3f CodeToPoint(
    const Code_t code, const float min_coord = 0.0f, const float range = 1.0f) {
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  uint_fast32_t dec_raw_x, dec_raw_y, dec_raw_z;
  libmorton::morton3D_64_decode(code, dec_raw_x, dec_raw_y, dec_raw_z);
  float dec_x = (static_cast<float>(dec_raw_x) / bit_scale) * range + min_coord;
  float dec_y = (static_cast<float>(dec_raw_y) / bit_scale) * range + min_coord;
  float dec_z = (static_cast<float>(dec_raw_z) / bit_scale) * range + min_coord;
  return {dec_x, dec_y, dec_z};
}
