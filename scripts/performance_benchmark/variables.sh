#!/usr/bin/env bash

TOOL=./tool.py

ALPHA=6e-5
BETA=0.42
MISSING=0.25

CELLS_SET="32 64 96 128"
CHAINS_SET="2"
STEPS_SET=`seq 50 50 200`
N_RUNS=4

BASE_DIR="performance_benchmark.out"