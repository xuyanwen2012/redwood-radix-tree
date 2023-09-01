cmake_minimum_required(VERSION 3.16)
project(my_bench LANGUAGES CXX)

add_executable(my_bench Morton.cpp)

target_link_libraries(my_bench PRIVATE my_lib)

# Add Google Benchmark
find_package(benchmark REQUIRED)
target_link_libraries(my_bench PRIVATE benchmark::benchmark)
