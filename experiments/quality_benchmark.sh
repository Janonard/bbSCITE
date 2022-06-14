#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter
#SBATCH -t 24:00:00
#SBATCH -J scite-quality-benchmark
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joo@mail.upb.de
#SBATCH -p normal
#SBATCH -q cont
#SBATCH -n 128
#SBATCH --ntasks-per-core 1

set -e 

module reset
module load fpga devel
module load intel/oneapi Boost CMake
source .venv/bin/activate

SCITE=./build/scite
FFSCITE=./build/src/ffSCITE_emu
TOOL=./tool.py

ALPHA=1e-6
BETA=0.25
MISSING=0.25

N_GENES=64
N_CELLS=64

REPETITIONS=10
LENGTH=1000000

function run_single_instance {
    EXEC=$1
    INPUT=$2
    OUT_DIR=$3

    $EXEC -i $INPUT -n $N_GENES -m $N_CELLS -r $REPETITIONS -l $LENGTH \
        -fd $ALPHA -ad $BETA -e 0 -move_probs 0.55 0.4 0.05 -o $OUT_DIR/tree -seed 42 \
        > $OUT_DIR/stdout.txt
    
    cat $OUT_DIR/stdout.txt | grep "Time elapsed" | cut -d" " -f3 > $OUT_DIR/runtime.txt

    $TOOL score -a $ALPHA -b $BETA -m $INPUT -t $OUT_DIR/tree_ml0.newick -x > $OUT_DIR/likelihood.txt
}

for n_cells in `seq 64 64 256`
do
    for n_genes in `seq 64 64 256`
    do
        BASE_DIR="out.${n_genes}x${n_cells}"
        mkdir -p $BASE_DIR

        for repetition in `seq 64`
        do
            REP_DIR="${BASE_DIR}/${repetition}"
            mkdir -p $REP_DIR

            $TOOL generate -n $N_GENES -m $N_CELLS -a $ALPHA -b $BETA -e $MISSING -o $REP_DIR
            INPUT="${REP_DIR}/input.csv"

            SCITE_DIR="${REP_DIR}/scite"
            FFSCITE_DIR="${REP_DIR}/ffSCITE"
            mkdir -p $SCITE_DIR $FFSCITE_DIR

            run_single_instance $SCITE $INPUT $SCITE_DIR &
            run_single_instance $FFSCITE $INPUT $FFSCITE_DIR &
        done

        wait
    done
done