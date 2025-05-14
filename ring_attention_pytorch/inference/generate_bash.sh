#!/bin/bash

# Model path and config
MODEL_PATH="$SCRATCH/models/Llama3.1-8B-Instruct/"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
CONFIG_FILE="3.1_8B.json"

# Loop over max_seq_len: 1024 -> 32768
for MAX_SEQ_LEN in 1024 2048 4096 8192 16384 32768
do
    echo "Running generation with max_seq_len = $MAX_SEQ_LEN"
    srun --gpus=1 torchrun --nproc_per_node=1 generate.py "$MODEL_PATH" "$MODEL_NAME" "$CONFIG_FILE" --max_seq_len "$MAX_SEQ_LEN"

    sleep 2
done

