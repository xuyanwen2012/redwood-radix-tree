#include <Eigen/Dense>
#include <boost/container/small_vector.hpp>
#include <cstdlib>
#include <iostream>

#include "Kernel.hpp"

int main() {
  Eigen::MatrixXd mat(2, 2);
  mat << 1, 2, 3, 4;

  Eigen::VectorXd vec(2);
  vec << 5, 6;

  Eigen::VectorXd result = mat * vec;

  std::cout << "Matrix:\n" << mat << "\n";
  std::cout << "Vector:\n" << vec << "\n";
  std::cout << "Result:\n" << result << "\n";

  Eigen::Vector3f a;
  boost::container::small_vector<uint32_t, 8> sv;

  // Add elements to the small_vector
  for (int i = 0; i < 4; ++i) {
    sv.push_back(i);
  }

  // Print the elements
  std::cout << "SmallVector elements: ";
  for (const auto& element : sv) {
    std::cout << element << " ";
  }
  std::cout << std::endl;

  // Print capacity and size
  std::cout << "Capacity: " << sv.capacity() << std::endl;
  std::cout << "Size: " << sv.size() << std::endl;

  TestKernel();

  return EXIT_SUCCESS;
}
