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
