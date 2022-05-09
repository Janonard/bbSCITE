set(CMAKE_CXX_COMPILER dpcpp)
cmake_minimum_required(VERSION 3.21)

project(ffSCITE CXX)
set(CMAKE_PROJECT_VERSION 0.1.0)

add_compile_options(-qactypes)

add_subdirectory(test)
