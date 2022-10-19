#!/usr/bin/env bash

set -e 

ml fpga devel intel/oneapi/22.2.0 bittware/520n Boost/1.74.0-GCC CMake

BASE_DIR=performance_benchmark.out
ALPHA=6e-5
BETA=0.42
MISSING=0.25

for CELLS in 16 32 64
do
    GENES=$(($CELLS-1))

    OUT_DIR=$BASE_DIR/$CELLS
    mkdir -p $OUT_DIR

    ./tool.py generate -o $OUT_DIR -n $GENES -m $CELLS -a $ALPHA -b $BETA -e $MISSING
done
