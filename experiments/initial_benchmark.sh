#!/usr/bin/env bash

set -e

SCITE=./build/scite
FFSCITE=./build/src/ffSCITE_emu
TOOL=./tool.py

ALPHA=1e-6
BETA=0.25
MISSING=0.25

N_GENES=64

function run_single_instance {
    EXEC=$1
    N_CELLS=$2
    LENGTH=$3
    BASE_DIR=$4
    EXEC_DIR=$5

    echo $LENGTH >> $EXEC_DIR/lengths.txt
    $EXEC -i $BASE_DIR/input.csv -n $N_GENES -m $N_CELLS -r 1 -l $LENGTH -fd $ALPHA -ad $BETA -e 0 -move_probs 0.55 0.4 0.05 -o $EXEC_DIR/tree_${LENGTH} -seed 42 \
        | grep "Time elapsed" | cut -d" " -f3 >> $EXEC_DIR/runtimes.txt
    $TOOL score -a $ALPHA -b $BETA -t $BASE_DIR/ -m $BASE_DIR/input.csv -t $EXEC_DIR/tree_${LENGTH}_ml0.newick -x >> $EXEC_DIR/likelihoods.txt
}

function run_benchmark {
    N_CELLS=$1

    BASE_DIR="${N_GENES}.out"
    SCITE_DIR="${BASE_DIR}/scite"
    FFSCITE_DIR="${BASE_DIR}/ffSCITE"
    mkdir -p $SCITE_DIR $FFSCITE_DIR

    $TOOL generate -n $N_GENES -m $N_CELLS -a $ALPHA -b $BETA -e $MISSING -o $BASE_DIR

    for LENGTH in `seq 100000 100000 1000000`
    do
        run_single_instance $SCITE $N_CELLS $LENGTH $BASE_DIR $SCITE_DIR &
        run_single_instance $FFSCITE $N_CELLS $LENGTH $BASE_DIR $FFSCITE_DIR &
        wait
    done
}

for N_CELLS in `seq 64 64 1024`
do
    run_benchmark $N_CELLS & 
done

wait