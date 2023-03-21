#!/usr/bin/env bash

set -e

source scripts/performance_benchmark/variables.sh

rm -rf $BASE_DIR
mkdir -p $BASE_DIR

for CELLS in $CELLS_SET
do
    GENES=$(($CELLS-1))

    OUT_DIR=$BASE_DIR/$CELLS
    mkdir -p $OUT_DIR

    ./tool.py generate -o $OUT_DIR -n $GENES -m $CELLS -a $ALPHA -b $BETA -e $MISSING &
done
wait
