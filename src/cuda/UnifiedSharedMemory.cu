#include <iostream>

#include "UnifiedSharedMemory.hpp"
#include "cuda/CudaUtils.cuh"

namespace redwood {

void* UsmMalloc(const std::size_t n) {
  void* tmp;
  HANDLE_ERROR(cudaMallocManaged(&tmp, n));
  std::cout << "accelerator::UsmMalloc() " << tmp << ": " << n << " bytes."
            << std::endl;
  return tmp;
}

void UsmFree(void* ptr) {
  std::cout << "accelerator::UsmFree() " << ptr << std::endl;
  if (ptr) {
    HANDLE_ERROR(cudaFree(ptr));
  }
}

}  // namespace redwood