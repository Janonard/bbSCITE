set(CMAKE_CXX_COMPILER dpcpp)
cmake_minimum_required(VERSION 3.21)

project(ffSCITE CXX)
set(CMAKE_PROJECT_VERSION 0.1.0)

find_package(Boost 1.74.0 REQUIRED)

add_compile_options(-qactypes -fintelfpga)

add_subdirectory(test)
add_subdirectory(src)
