#!/usr/bin/env bash

module reset
module load fpga devel intel/oneapi bittware/520n/20.4.0_hpc Boost/1.74.0-GCC CMake

SCITE=./build/scite
FFSCITE=./build/ffSCITE
TOOL=./tool.py

ALPHA=1e-6
BETA=0.25
MISSING=0.25

N_GENES=16
N_CELLS=16

N_INPUTS=64
REPETITIONS=5
LENGTH=1000000

BASE_DIR="quality_benchmark.out"