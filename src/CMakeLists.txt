cmake_minimum_required(VERSION 3.16)
project(my_lib LANGUAGES CXX CUDA)

add_library(my_lib)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/*.hpp")

target_sources(
  my_lib
  PRIVATE cuda/BinaryRadixTree.cu cuda/MortonKernels.cu cuda/Octree.cu
          cuda/PrefixSum.cu cuda/UnifiedSharedMemory.cu
  PUBLIC ${HEADER_LIST})

target_include_directories(
  my_lib
  PUBLIC ${CMAKE_SOURCE_DIR}/extern/cxxopts/include
  PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_features(my_lib PUBLIC cxx_std_14)

# Add Eigen
find_package(Eigen3 REQUIRED)
target_include_directories(my_lib PUBLIC ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
# Ok so using Eigen + CUDA has a bunch of warnings, so we suppress them
target_compile_options(
  my_lib PRIVATE -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)

# CUDA

target_compile_options(my_lib PRIVATE -arch=sm_75 --expt-relaxed-constexpr)

set_target_properties(my_lib PROPERTIES CUDA_ARCHITECTURES 87)

if(NOT WIN32)
  # Currently there is a bug causing linker error on Windows if separable
  # compilation is enabled
  set_target_properties(my_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(my_lib
                           PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(my_lib PUBLIC ${CUDA_LIBRARIES})

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(my_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
