#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter
#SBATCH -p fpga
#SBATCH --constraint=bittware_520n_20.4.0_hpc
#SBATCH --time=24:00:00
#SBATCH --mail-type=All
#SBATCH --mail-user=joo@mail.upb.de

set -e 

module reset
module load fpga devel intel/oneapi bittware/520n/20.4.0_hpc Boost/1.74.0-GCC CMake

SCITE=./build/scite
FFSCITE=./build/ffSCITE
TOOL=./tool.py

ALPHA=1e-6
BETA=0.25
MISSING=0.25

N_GENES=16
N_CELLS=16

N_INPUTS=64
REPETITIONS=5
LENGTH=1000000

BASE_DIR="quality_benchmark.out"
rm -rf $BASE_DIR
mkdir -p $BASE_DIR

# Generate inputs
for input_i in `seq $N_INPUTS`
do
    INPUT_DIR="${BASE_DIR}/${input_i}"
    mkdir -p $INPUT_DIR

    $TOOL generate -n $N_GENES -m $N_CELLS -a $ALPHA -b $BETA -e $MISSING -o $INPUT_DIR
done
wait

echo "Inputs generated"

function run {
    EXEC=$1
    EXEC_ID=$2

    for input_i in `seq $N_INPUTS`
    do
        INPUT="${BASE_DIR}/${input_i}/input.csv"
        OUT_DIR="${BASE_DIR}/${input_i}/${EXEC_ID}/"
        mkdir -p $OUT_DIR

        # Run the simulation once.
        $EXEC -i $INPUT -n $N_GENES -m $N_CELLS -r $REPETITIONS -l $LENGTH -fd $ALPHA -ad $BETA -e 0 -move_probs 0.55 0.4 0.05 -o $OUT_DIR/tree -max_treelist_size 1 > $OUT_DIR/stdout.txt \
            && $TOOL score -a $ALPHA -b $BETA -t "$OUT_DIR/tree_ml0.newick" -m $INPUT -x > $OUT_DIR/likelihood.txt \
            && cat $OUT_DIR/stdout.txt | grep "Time elapsed" | cut -d" " -f3 > $OUT_DIR/runtime.txt
    done
    wait

    echo "${EXEC_ID} finished the simulation"
}

run $FFSCITE "ffSCITE"
run $SCITE "SCITE"
