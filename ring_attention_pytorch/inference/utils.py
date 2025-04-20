import torch


def collate(inputs: list[list[int]], device: torch.device | str):
    batch_size = len(inputs)
    max_prompt_len = max([len(p) for p in inputs])

    padded = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(
        batch_size, max_prompt_len, dtype=torch.bool, device=device
    )
    input_pos = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)

    for i, e in enumerate(inputs):
        prompt_len = len(e)
        input_pos[i, -prompt_len:] = torch.arange(prompt_len, device=device)
        padded[i, -prompt_len:] = torch.tensor(e, dtype=torch.long, device=device)
        attention_mask[i, -prompt_len:] = True

    return padded, attention_mask, input_pos


def sample(logits: torch.Tensor, temperature=1.0):
    pass
