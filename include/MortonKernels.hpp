#pragma once

#include <Eigen/Dense>

#include "Morton.hpp"

void ComputeMortonCodes(const Eigen::Vector3f* inputs, Code_t* morton_keys,
                        size_t n, float min_coord = 0.0f, float range = 1.0f);

void SortMortonCodes(const Code_t* morton_keys, Code_t* sorted_morton_keys,
                     size_t n);
