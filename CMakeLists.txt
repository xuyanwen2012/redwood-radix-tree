cmake_minimum_required(VERSION 3.16)
project(my_app LANGUAGES CXX CUDA)

add_executable(my_app)

# file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/*.hpp")

target_sources(my_app
  PRIVATE
    src/Main.cpp
    src/Octree.cpp
    src/BinaryRadixTree.cpp
)

target_include_directories(my_app 
    PRIVATE libmorton/include
    PUBLIC include
)

target_compile_features(my_app PUBLIC cxx_std_17)
target_compile_options(my_app PRIVATE -Wall)

# Add Eigen
find_package(Eigen3 REQUIRED)
target_include_directories(my_app PUBLIC ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

# # Add Boost
# find_package(Boost REQUIRED COMPONENTS container)
# message(STATUS "Boost version: ${Boost_VERSION}")
# target_link_libraries(my_app PUBLIC Boost::container)

# CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 --expt-relaxed-constexpr")

set_target_properties(my_app 
  PROPERTIES 
    CUDA_ARCHITECTURES 61 75 87
    CUDA_SEPARABLE_COMPILATION ON
)
        
target_include_directories(my_app PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(my_app PUBLIC ${CUDA_LIBRARIES})
