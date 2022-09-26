#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter
#SBATCH -p fpga
#SBATCH --constraint=bittware_520n_20.4.0_hpc
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --mail-type=All
#SBATCH --mail-user=joo@mail.upb.de

set -e 

source scripts/quality_benchmark/variables.sh
EXEC="./build/ffSCITE"
EXEC_ID="ffSCITE"

for input_i in `seq $N_INPUTS`
do
    INPUT="${BASE_DIR}/${input_i}/input.csv"
    OUT_DIR="${BASE_DIR}/${input_i}/${EXEC_ID}/"
    mkdir -p $OUT_DIR

    $EXEC -i $INPUT -n $N_GENES -m $N_CELLS -r $REPETITIONS -l $LENGTH -fd $ALPHA -ad $BETA -e 0 -move_probs 0.55 0.4 0.05 -o $OUT_DIR/tree -max_treelist_size 1 > $OUT_DIR/stdout.txt \
        && $TOOL score -a $ALPHA -b $BETA -t "$OUT_DIR/tree_ml0.newick" -m $INPUT -x > $OUT_DIR/likelihood.txt \
        && cat $OUT_DIR/stdout.txt | grep "Time elapsed" | cut -d" " -f3 > $OUT_DIR/runtime.txt
done
wait

echo "${EXEC_ID} finished the simulation"
