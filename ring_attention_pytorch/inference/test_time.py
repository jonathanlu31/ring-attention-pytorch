# This script can be used to compare the runtimes of ring attention, striped ring attention, flash attention, and eager attention.

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributed import get_world_size

from ring_attention_pytorch.inference.attention_variants import RingAttentionLlama, Attention

@torch.inference_mode()
def benchmark_attention(model, name, input):
    for _ in range(10):
        _ = model.forward_attention(input)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.time()
        _ = model.forward_attention(input)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    print(f"{name}: {avg_time * 1000:.3f} ms")
    return avg_time

if __name__ == "__main__":
    print("Timing attention modules...")

    batch_size = 4
    seq_len = 512
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    # device = "cuda"
    device = "cpu"
    num_iters = 100

    ring_attention = RingAttentionLlama(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dim=dim,
        # bucket_size=512,      # for some reason this is in custom_model.py but idt the arg exists
        ring_seq_size=512,
        ring_size=get_world_size(),
        use_striped=False,
    ).to(device)

    striped_ring_attention = RingAttentionLlama(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dim=dim,
        # bucket_size=512,        # see above
        ring_seq_size=512,
        ring_size=get_world_size(),
        use_striped=True,
    ).to(device)

    eager_attention = Attention(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dim=dim,
        rotary_embed=True,
        max_seq_len=seq_len,
    ).to(device)

    input = torch.randn(batch_size, seq_len, dim, device=device)

    benchmark_attention(ring_attention, "Ring Attention", input)
    benchmark_attention(striped_ring_attention, "Striped Ring Attention", input)
    # benchmark_attention(flash_attention, "Flash Attention", input, start_pos, freqs_cis, mask)    # todo?
    benchmark_attention(eager_attention, "Eager Attention", input)
