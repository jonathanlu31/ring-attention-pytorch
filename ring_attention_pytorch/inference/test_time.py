# This script can be used to compare the runtimes of ring attention, striped ring attention, flash attention, and eager attention.
# Example: python test_time.py --runtimes ring striped_ring

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributed import get_world_size
import argparse

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
    parser = argparse.ArgumentParser(description="Benchmark attention variants")
    parser.add_argument(
        "--runtimes", 
        type=str, 
        nargs="+", 
        default=["ring", "striped_ring", "flash", "eager"],
        help="Which attention runtimes to benchmark. Options: ring, striped_ring, flash, eager (all by default)"
    )
    args = parser.parse_args()
    print("Timing attention modules...")

    batch_size = 4
    seq_len = 512
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    # device = "cuda"
    device = "cpu"
    num_iters = 100

    models = {}

    if "ring" in args.runtimes:
        models["Ring Attention"] = RingAttentionLlama(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dim=dim,
            # bucket_size=512,      # for some reason this is in custom_model.py but idt the arg exists
            ring_seq_size=512,
            ring_size=get_world_size(),
            use_striped=False,
        ).to(device)

    if "striped_ring" in args.runtimes:
        models["Striped Ring Attention"] = RingAttentionLlama(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dim=dim,
            # bucket_size=512,        # see above
            ring_seq_size=512,
            ring_size=get_world_size(),
            use_striped=True,
        ).to(device)

    if "eager" in args.runtimes:
        models["Eager Attention"] = Attention(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dim=dim,
            rotary_embed=True,
            max_seq_len=seq_len,
            use_flash=False
        ).to(device)

    if "flash" in args.runtimes:
        models["Flash Attention"] = Attention(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dim=dim,
            rotary_embed=True,
            max_seq_len=seq_len,
            use_flash=True
        ).to(device)

    input = torch.randn(batch_size, seq_len, dim, device=device)

    for name, model in models.items():
        benchmark_attention(model, name, input, num_iters)
