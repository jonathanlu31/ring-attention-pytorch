## Usage

For `generate.py`:
```bash
torchrun --nproc_per_node 3 generate.py /data/jonathan/models/llama-models/Llama-3.1-8B-Instruct/ meta-llama/Llama-3.1-8B-Instruct 3.1_8B_ring.json
```

The first argument is the path to checkpoint directory for the Llama weights. The second argument is the tokenizer path for HuggingFace. The third is any of the params files in ring_llama_params. 

For normal, no sequence parallelism, you'd run:
```bash
torchrun --nproc_per_node 1 generate.py /data/jonathan/models/llama-models/Llama-3.1-8B-Instruct/ meta-llama/Llama-3.1-8B-Instruct 3.1_8B.json
```

## Testing

Test end to end transformer forward pass
```bash
python run_causal_transformer.py --param-file 3.1_8B.json --output-file 3.1_8B.pt
python run_causal_transformer.py --param-file 3.1_8B_ring.json --output-file 3.1_8B_ring.pt --world-size 2
python check_correctness_transformer.py --file-a 3.1_8B.pt --file-b 3.1_8B_ring.pt
```

Test causal attention
```bash
python assert_causal_attn.py --world-size 2 --batch-size 2
```

# Numbers:

BS 8, prompt len 20, total generated tokens 4096
Flash: 26.3 tokens per second
Flash + cache: 138 tokens per second
Ring 1 GPU: 27.1
Ring 2 GPU: 23.6
Ring 4 GPU: 16.8
Ring 1 GPU cache:
Ring 2 GPU cache:
Ring 4 GPU cache:
Ring 1 GPU cache + fast:
Ring 2 GPU cache + fast: 70.6
Ring 4 GPU cache + fast: 70.3
