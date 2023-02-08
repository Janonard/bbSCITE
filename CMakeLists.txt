set(CMAKE_CXX_COMPILER dpcpp)
set(CXX_STANDARD 20)
cmake_minimum_required(VERSION 3.21)

project(ffSCITE CXX)
set(CMAKE_PROJECT_VERSION 0.1.0)

find_package(Boost 1.79.0 REQUIRED)

add_subdirectory(cpu_scoring_benchmark)
add_subdirectory(test)
add_subdirectory(src)

set(SCITE_SOURCES
  SCITE/findBestTrees.cpp
  SCITE/mcmcBinTreeMove.cpp
  SCITE/mcmcTreeMove.cpp
  SCITE/rand.cpp
  SCITE/scoreTree.cpp
  SCITE/trees.cpp
  SCITE/matrices.cpp
  SCITE/mcmc.cpp
  SCITE/output.cpp
  SCITE/scoreBinTree.cpp
  SCITE/treelist.cpp)

add_executable(scite ${SCITE_SOURCES})