#!/usr/bin/env bash

TOOL=./tool.py

ALPHA=6e-5
BETA=0.42
MISSING=0.25

CELLS_SET="32 64 95"
CHAINS_SET="24 48"
STEPS_SET=`seq 500000 500000 2000000`
N_RUNS=4

BASE_DIR="performance_benchmark.out"