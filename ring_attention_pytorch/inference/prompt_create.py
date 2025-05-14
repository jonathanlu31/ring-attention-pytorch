from transformers import AutoTokenizer
import json

# 1. Load the tokenizer (replace with your actual tokenizer path if local)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 2. Load Shakespeare text
with open("/pscratch/sd/a/andre_g/prompts/shakespeare.txt", "r") as f:
    text = f.read()

# 3. Function to build long prompt with token limit
def make_long_prompt(base_text, target_tokens):
    full_text = base_text
    while True:
        num_tokens = len(tokenizer.encode(full_text, truncation=False))
        if num_tokens >= target_tokens:
            break
        full_text += "\n\n" + base_text

    # Truncate to exactly target_tokens
    tokens = tokenizer.encode(full_text, truncation=False)
    full_text = tokenizer.decode(tokens[:target_tokens])
    return full_text

# 4. Generate and save prompts of different lengths
for target_len in [32000, 50000, 64000, 75000, 100000]:
    repeated_text = make_long_prompt(text, target_len)
    instruction = "\n\nRepeat 10 to 100 words from within the text."
    prompt = repeated_text + instruction

    # Save as JSONL
    with open(f"prompt_{target_len}_tokens.jsonl", "w") as f:
        f.write(json.dumps({ "role": "user", "content": prompt }) + "\n")

