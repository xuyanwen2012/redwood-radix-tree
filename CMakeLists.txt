cmake_minimum_required(VERSION 3.12)
project(my_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE CXX_SOURCES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${SRC_DIR}/*.cu")

add_executable(my_app ${CXX_SOURCES} ${CU_SOURCES})
target_include_directories(my_app PRIVATE libmorton/include)

# Add Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add Boost
find_package(Boost REQUIRED COMPONENTS container)
message(STATUS "Boost version: ${Boost_VERSION}")

# CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 --expt-relaxed-constexpr")

set_target_properties(my_app PROPERTIES CUDA_ARCHITECTURES 61 75 87)
set_target_properties(my_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include_directories(
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(my_app PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(my_app PUBLIC Boost::container)
