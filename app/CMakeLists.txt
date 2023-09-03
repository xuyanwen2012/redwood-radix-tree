cmake_minimum_required(VERSION 3.16)
project(MyApp LANGUAGES CXX)

add_executable(MyApp Main.cpp)

target_link_libraries(MyApp PRIVATE my_lib)
