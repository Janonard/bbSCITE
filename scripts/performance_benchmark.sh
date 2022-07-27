#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter
#SBATCH -p fpga
#SBATCH --constraint=bittware_520n_20.4.0_hpc
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --mail-type=All
#SBATCH --mail-user=joo@mail.upb.de

set -e 

ml fpga devel intel/oneapi bittware/520n Boost/1.74.0-GCC CMake

EXEC=./build/ffSCITE
INPUT=./SCITE/dataHou18.csv
OUT_DIR=performance_benchmark.out

mkdir -p $OUT_DIR

for N_CHAINS in `seq 2`
do
    for N_STEPS in `seq 50000 50000 250000`
    do
        for i in `seq 10`
        do
            $EXEC -i $INPUT -r $N_CHAINS -l $N_STEPS -fd 6e-5 -ad 0.42 \
                | grep "Time elapsed" \
                | cut -d" " -f3 \
                >> "${OUT_DIR}/${N_CHAINS}_${N_STEPS}.log"
        done
    done
done
