cmake_minimum_required(VERSION 3.12)
project(my_app LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE CXX_SOURCES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE CU_SOURCES "${SRC_DIR}/*.cu")

add_executable(${PROJECT_NAME} ${CXX_SOURCES} ${CU_SOURCES})

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR} 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(my_app PRIVATE ${CUDA_LIBRARIES})
