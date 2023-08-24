#include <Eigen/Dense>
#include <algorithm>
#include <bitset>
#include <boost/container/small_vector.hpp>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

#include "BinaryRadixTree.hpp"
#include "Common.hpp"
#include "Morton.hpp"

void PrintVector3F(const Eigen::Vector3f& vec) {
  std::cout << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")\n";
}

int main() {
  thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution dis(0.0f, 1.0f);

  // Prepare Inputs
  constexpr int n = 64;
  std::vector<Eigen::Vector3f> inputs(n);
  std::generate(inputs.begin(), inputs.end(),
                [&] { return Eigen::Vector3f(dis(gen), dis(gen), dis(gen)); });

  // std::for_each(inputs.begin(), inputs.end(), PrintVector3F);

  // [Step 1] Compute Morton Codes
  std::vector<Code_t> morton_keys;
  morton_keys.reserve(n);
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(morton_keys),
                 [&](const auto& vec) {
                   const auto x = vec.x();
                   const auto y = vec.y();
                   const auto z = vec.z();
                   return MortonCode32(ToNBitInt(x), ToNBitInt(y),
                                       ToNBitInt(z));
                 });

  // [Step 2] Sort Morton Codes by Key
  std::sort(morton_keys.begin(), morton_keys.end());

  // [Step 3-4] Handle Duplicates
  morton_keys.erase(std::unique(morton_keys.begin(), morton_keys.end()),
                    morton_keys.end());

  std::for_each(morton_keys.begin(), morton_keys.end(), [](const auto key) {
    std::cout << key << "\t" << std::bitset<32>(key) << "\t" << std::endl;
  });

  // [Step 5] Build Binary Radix Tree
  constexpr auto num_brt_nodes = n - 1;
  std::vector<brt::InnerNodes> inners(num_brt_nodes);

  for (int i = 0; i < num_brt_nodes; ++i) {
    brt::MyProcessInternalNode(n, morton_keys.data(), i, inners.data());
  }

  for (int i = 0; i < num_brt_nodes; ++i) {
    std::cout << "Node " << i << "\n";
    std::cout << "\tdelta_node: " << inners[i].delta_node << "\n";
    std::cout << "\tsfc_code: " << inners[i].sfc_code << "\n";
    std::cout << "\tleft: " << inners[i].left << "\n";
    std::cout << "\tright: " << inners[i].right << "\n";
    std::cout << "\tparent: " << inners[i].parent << "\n";
    std::cout << "\n";
  }

  return EXIT_SUCCESS;
}
