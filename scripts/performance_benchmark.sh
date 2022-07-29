#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter
#SBATCH -p fpga
#SBATCH --constraint=bittware_520n_20.4.0_hpc
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=All
#SBATCH --mail-user=joo@mail.upb.de

set -e 

ml fpga devel intel/oneapi bittware/520n Boost/1.74.0-GCC CMake

OUT_DIR=performance_benchmark.out
FFSCITE_DIR=$OUT_DIR/ffSCITE
SCITE_DIR=$OUT_DIR/SCITE

mkdir -p $OUT_DIR $FFSCITE_DIR $SCITE_DIR

GENES=63
CELLS=64
ALPHA=6e-5
BETA=0.42
MISSING=0.25

./tool.py generate -o $OUT_DIR -n $GENES -m $CELLS -a $ALPHA -b $BETA -e $MISSING

INPUT=$OUT_DIR/input.csv

for N_CHAINS in `seq 3 1 5`
do
    for N_STEPS in `seq 50000 50000 400000`
    do
        for i in `seq 10`
        do
            ./build/ffSCITE \
                -i $INPUT -r $N_CHAINS -l $N_STEPS -fd $ALPHA -ad $BETA -max_treelist_size 1 \
                >> "${FFSCITE_DIR}/${N_CHAINS}_${N_STEPS}.log"

            ./build/scite \
                -n $GENES -m $CELLS -i $INPUT -r $N_CHAINS -l $N_STEPS -fd $ALPHA -ad $BETA -max_treelist_size 1 \
                >> "${SCITE_DIR}/${N_CHAINS}_${N_STEPS}.log"
        done
    done
done
