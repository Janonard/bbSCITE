#!/usr/bin/env bash

set -e 

SCITE=$1
FFSCITE=$2
TOOL=./tool.py

ALPHA=1e-6
BETA=0.25
MISSING=0.25

N_GENES=64
N_CELLS=64

N_INPUTS=128
REPETITIONS=10
LENGTH=1000000

BASE_DIR="quality_benchmark.out"
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

function run {
    EXEC=$1
    EXEC_ID=$2

    # For every repetition,
    for rep_i in `seq $REPETITIONS`
    do
        # and every input,
        for input_i in `seq $N_INPUTS`
        do
            INPUT="${BASE_DIR}/${input_i}/input.csv"
            OUT_DIR="${BASE_DIR}/${input_i}/${EXEC_ID}/${rep_i}/"
            mkdir -p $OUT_DIR

            # Run the simulation once.
            $EXEC -i $INPUT -n $N_GENES -m $N_CELLS -r 1 -l $LENGTH -fd $ALPHA -ad $BETA -e 0 -move_probs 0.55 0.4 0.05 -o $OUT_DIR/tree > $OUT_DIR/stdout.txt \
                && $TOOL score -a $ALPHA -b $BETA -t "$OUT_DIR/tree_ml0.newick" -m $INPUT -x > $OUT_DIR/likelihood.txt \
                && cat $OUT_DIR/stdout.txt | grep "Time elapsed" | cut -d" " -f3 > $OUT_DIR/runtime.txt &
        done
        wait

        echo "${EXEC_ID} completed repetition ${rep_i} at $(date)"
    done

    echo "${EXEC_ID} finished the simulation"
}

run $FFSCITE "ffSCITE"
run $SCITE "SCITE"
