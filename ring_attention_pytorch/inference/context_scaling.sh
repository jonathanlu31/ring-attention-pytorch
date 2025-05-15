#!/bin/bash

# Model path and config
MODEL_PATH="$SCRATCH/models/Llama3.1-8B-Instruct/"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
CONFIG_FILE="3.1_8B.json"

# Loop over context_len values
for CONTEXT_LEN in 32000 50000 64000 75000 100000
do
    echo "Running generation with context_len = $CONTEXT_LEN"
    srun --gpus=1 torchrun --nproc_per_node=1 generate.py "$MODEL_PATH" "$MODEL_NAME" "$CONFIG_FILE" --context_len "$CONTEXT_LEN"

    sleep 2
done

