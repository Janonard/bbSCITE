#!/usr/bin/env bash

set -e 

source scripts/quality_benchmark/variables.sh

rm -rf $BASE_DIR
mkdir -p $BASE_DIR

# Generate inputs
for input_i in `seq $N_INPUTS`
do
    INPUT_DIR="${BASE_DIR}/${input_i}"
    mkdir -p $INPUT_DIR

    $TOOL generate -n $N_GENES -m $N_CELLS -a $ALPHA -b $BETA -e $MISSING -o $INPUT_DIR &
done
wait

echo "Inputs generated"