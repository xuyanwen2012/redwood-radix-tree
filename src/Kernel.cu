#include <iostream>

#include "Kernel.hpp"

__global__ void helloCUDA() {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Hello, CUDA! This is thread %d.\n", threadId);
}

void TestKernel() {
  int numThreadsPerBlock = 256;
  int numBlocks = 4;

  helloCUDA<<<numBlocks, numThreadsPerBlock>>>();

  cudaDeviceSynchronize();
}
