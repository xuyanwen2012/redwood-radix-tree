cmake_minimum_required(VERSION 3.16)
project(my_app LANGUAGES CXX CUDA)

add_executable(my_app)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/*.hpp")

target_sources(
  my_app
  PRIVATE src/Main.cpp src/Octree.cpp src/BinaryRadixTree.cpp
  PUBLIC ${HEADER_LIST})

target_include_directories(
  my_app
  PRIVATE ${CMAKE_SOURCE_DIR}/extern/libmorton/include
          ${CMAKE_SOURCE_DIR}/extern/cxxopts/include
  PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_features(my_app PUBLIC cxx_std_17)
target_compile_options(my_app PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Add Eigen
find_package(Eigen3 REQUIRED)
target_include_directories(my_app PUBLIC ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

# CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 --expt-relaxed-constexpr")

set_target_properties(my_app PROPERTIES CUDA_ARCHITECTURES 87
                                        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(my_app
                           PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(my_app PUBLIC ${CUDA_LIBRARIES})
