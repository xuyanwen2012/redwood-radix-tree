cmake_minimum_required(VERSION 3.16)
project(MyApp LANGUAGES CXX)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/*.hpp")

add_executable(MyApp ${HEADER_LIST} Main.cpp)

target_compile_features(MyApp PRIVATE cxx_std_17)

target_include_directories(MyApp PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(MyApp PRIVATE my_lib)
