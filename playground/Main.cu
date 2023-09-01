

#include "Morton.hpp"

int main() {
  constexpr float min_coord = 0.0f;
  constexpr float range = 10.0f;
  std::cout << PointToCode(1, 2, 3, min_coord, range) << std::endl;
  std::cout << MyVersionPointToCode(1, 2, 3, min_coord, range) << std::endl;

  return 0;
}