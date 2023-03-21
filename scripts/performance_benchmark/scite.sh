#!/usr/bin/env bash

set -e 

source scripts/performance_benchmark/variables.sh

for CELLS in $CELLS_SET
do
    GENES=$(($CELLS-1))

    OUT_DIR=$BASE_DIR/$CELLS
    SCITE_DIR=$OUT_DIR/SCITE
    mkdir -p $SCITE_DIR

    INPUT=$OUT_DIR/input.csv

    for N_CHAINS in $CHAINS_SET
    do
        for N_STEPS in $STEPS_SET
        do
            for i in `seq $N_RUNS`
            do
                ./build/scite \
                    -n $GENES -m $CELLS -i $INPUT -r $N_CHAINS -l $N_STEPS -fd $ALPHA -ad $BETA -max_treelist_size 1 \
                    >> "${SCITE_DIR}/${N_CHAINS}_${N_STEPS}.log"
            done &
        done
    done
done
wait
