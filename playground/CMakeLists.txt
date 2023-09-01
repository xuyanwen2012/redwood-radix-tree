cmake_minimum_required(VERSION 3.16)
project(Playground LANGUAGES CXX)

add_executable(Playground Main.cu UnifiedSharedMemory.cu)

target_link_libraries(Playground PRIVATE my_lib)

# CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 --expt-relaxed-constexpr")

set_target_properties(Playground PROPERTIES CUDA_ARCHITECTURES 87
                                            CUDA_SEPARABLE_COMPILATION ON)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(Playground PUBLIC OpenMP::OpenMP_CXX)
endif()
