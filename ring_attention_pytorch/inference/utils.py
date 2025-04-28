import torch


def collate(inputs: list[list[int]] | list[torch.Tensor], device: torch.device | str):
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

def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1).to(dtype=torch.int64)

def logits_to_probs(logits, temperature: float = 1.0, top_k: int | None = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: int | None = None):
    if temperature == 0:
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True)

    # Check this
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next
