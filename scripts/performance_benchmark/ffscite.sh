#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter
#SBATCH -p fpga
#SBATCH --constraint=bittware_520n_20.4.0_hpc
#SBATCH --time=02:00:00
#SBATCH --mail-type=All
#SBATCH --mail-user=joo@mail.upb.de

set -e 

ml fpga devel intel/oneapi/22.2.0 bittware/520n Boost/1.79.0-GCC CMake

BASE_DIR=performance_benchmark.out
ALPHA=6e-5
BETA=0.42
MISSING=0.25

for CELLS in 16 32 64
do
    GENES=$(($CELLS-1))

    OUT_DIR=$BASE_DIR/$CELLS
    FFSCITE_DIR=$OUT_DIR/ffSCITE
    SCITE_DIR=$OUT_DIR/SCITE
    mkdir -p $FFSCITE_DIR $SCITE_DIR

    INPUT=$OUT_DIR/input.csv

    for N_CHAINS in 24 48
    do
        for N_STEPS in `seq 500000 500000 2000000`
        do
            for i in `seq 4`
            do
                ./build/ffSCITE \
                    -i $INPUT -r $N_CHAINS -l $N_STEPS -fd $ALPHA -ad $BETA -max_treelist_size 1 \
                    >> "${FFSCITE_DIR}/${N_CHAINS}_${N_STEPS}.log"
            done
        done
    done
done
