#!/bin/bash

# Ring runtime with world size = 2
WORLD_SIZE=2
RUNTIME="ring"

# Loop over sequence lengths: 1024 -> 131072 (powers of 2)
for SEQ_LEN in 1024 2048 4096 8192 16384 32768 65536 131072
do
    echo "Running benchmark with seq_len = $SEQ_LEN"
    python benchmark_attention.py --runtime $RUNTIME --world-size $WORLD_SIZE --seq_len $SEQ_LEN

    sleep 2
done
