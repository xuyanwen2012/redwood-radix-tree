#pragma once

#include <vector>

namespace redwood {

/**
 * @brief Malloc Unified Shared Memory that is shared between CPU host and GPU
 * device. This is effective on e.g., Nvidia Jetson or Intel UHD Graphics.
 *
 * @param n size in bytes
 * @return void* pointer to the allocated memory
 */
void* UsmMalloc(std::size_t n);

/**
 * @brief Free Unified Shared Memory that is shared between CPU and GPU
 *
 * @param ptr pointer to the memory to be freed
 */
void UsmFree(void* ptr);

/**
 * @brief Malloc Unified Shared Memory that is shared between CPU host and GPU.
 *
 * @tparam T type of the elements
 * @param n number of elements
 * @return T* pointer to the allocated memory
 */
template <typename T>
T* UsmMalloc(const std::size_t n) {
  return static_cast<T*>(UsmMalloc(n * sizeof(T)));
}

/**
 * @brief Allocator for Unified Shared Memory that is shared between CPU host.
 * You can use it with std::vector. See UsmVector.
 *
 * @tparam T type of the elements
 */
template <typename T>
class UsmAlloc {
 public:
  // must not change name
  using value_type = T;
  using pointer = value_type*;

  UsmAlloc() noexcept = default;

  template <typename U>
  UsmAlloc(const UsmAlloc<U>&) noexcept {}

  // must not change name
  value_type* allocate(std::size_t n, const void* = nullptr) {
    return static_cast<value_type*>(UsmMalloc(n * sizeof(value_type)));
  }

  void deallocate(pointer p, std::size_t n) {
    if (p) {
      UsmFree(p);
    }
  }
};

/* Equality operators */
template <class T, class U>
bool operator==(const UsmAlloc<T>&, const UsmAlloc<U>&) {
  return true;
}

template <class T, class U>
bool operator!=(const UsmAlloc<T>&, const UsmAlloc<U>&) {
  return false;
}

template <typename T>
using UsmVector = std::vector<T, redwood::UsmAlloc<T>>;
}  // namespace redwood
