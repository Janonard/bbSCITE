#!/usr/bin/env bash

source scripts/performance_benchmark/variables.sh

for VECTORLEN in 64 96
do
    for CELLS in $CELLS_SET
    do
        GENES=$(($CELLS-1))
        OUT_DIR=$BASE_DIR/$CELLS
        INPUT=$OUT_DIR/input.csv

        if [ $VECTORLEN -lt $CELLS ]; then
            continue
        fi

        echo "$VECTORLEN $CELLS"

        FFSCITE_DIR=$OUT_DIR/bbSCITE$VECTORLEN
        mkdir -p $FFSCITE_DIR
        FFSCITE=./build/src/bbSCITE${VECTORLEN}

        for N_CHAINS in $CHAINS_SET
        do
            for N_STEPS in $STEPS_SET
            do
                LOGFILE="${FFSCITE_DIR}/${N_CHAINS}_${N_STEPS}.log"
                for i in `seq $N_RUNS`
                do
                    $FFSCITE \
                        -i $INPUT -r $N_CHAINS -l $N_STEPS -fd $ALPHA -ad $BETA -max_treelist_size 1 \
                        >> $LOGFILE &

                    while [ `jobs -r | wc -l` -gt 0 ]
                    do
                        newgrp dialout <<< /usr/share/nallatech/520n/bist/utilities/nalla_serial_cardmon/bin/nalla_serial_cardmon \
                        | grep "Total board power" >> $LOGFILE
                        echo "At instant $(date -Iseconds)" >> $LOGFILE
                    done

                    wait # Should not be necessary, but there for contingency.
                done
            done
        done
    done
done
