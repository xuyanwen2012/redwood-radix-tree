cmake_minimum_required(VERSION 3.12)
project(my_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE CXX_SOURCES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${SRC_DIR}/*.cu")

# Add Eigen
find_package(Eigen3 REQUIRED)

# Add Boost
find_package(Boost REQUIRED COMPONENTS container)
message(STATUS "Boost version: ${Boost_VERSION}")

# CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 --expt-relaxed-constexpr")

add_executable(my_app ${CXX_SOURCES} ${CU_SOURCES})

set_target_properties(my_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(my_app PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(my_app PUBLIC Boost::container)
