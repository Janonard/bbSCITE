#!/usr/bin/env bash

module reset
module load fpga devel intel/oneapi/22.2.0 bittware/520n/20.4.0_hpc Boost/1.79.0-GCC CMake

TOOL=./tool.py

ALPHA=1e-6
BETA=0.25
MISSING=0.25

N_GENES=127
N_CELLS=128

N_INPUTS=64
REPETITIONS=24
LENGTH=500000

BASE_DIR="quality_benchmark.out"