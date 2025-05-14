from transformers import AutoTokenizer
import random

# Load the tokenizer (change to match your model)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Load Shakespeare text
with open("/pscratch/sd/a/andre_g/prompts/shakespeare.txt", "r") as f:
    text = f.read()

# Function to pad text to desired token length
def make_long_prompt(base_text, target_tokens):
    full_text = base_text
    while True:
        num_tokens = len(tokenizer.encode(full_text, truncation=False))
        if num_tokens >= target_tokens:
            break
        full_text += "\n\n" + base_text
    return full_text

# Generate prompts at various lengths
for target_len in [32000, 50000, 75000, 100000]:
    repeated_text = make_long_prompt(text, target_len)

    # Add instruction at the end
    instruction = "\n\nRepeat 10 to 100 words from within the text."
    prompt = repeated_text + instruction

    # Save to file or jsonl
    with open(f"prompt_{target_len}_tokens.jsonl", "w") as f:
        f.write(f'{ { "role": "user", "content": prompt } }\n')

