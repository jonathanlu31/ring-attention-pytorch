# This script can be used to compare the runtimes of ring attention, striped ring attention, flash attention, and eager attention.
# Example: python test_time.py --runtime ring striped_ring

import time
import torch
import torch.multiprocessing as mp
import argparse
import os
import torch.distributed as dist

import torch.profiler

from ring_attention_pytorch.inference.attention_variants import (
    RingAttentionLlama,
    Attention,
)
from ring_attention_pytorch.inference.distributed import cleanup


def setup(
    rank,
    world_size,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    backend = "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


@torch.inference_mode()
def benchmark_attention(
    rank,
    world_size,
    batch_size,
    seq_len,
    attn_implementation,
    dim,
    n_heads,
    n_kv_heads,
    dtype,
    num_iters,
    profile=False
):
    setup(rank, world_size)
    print(f"rank {rank} started")

    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        dtype = getattr(torch, dtype)
        input = torch.randn(batch_size, seq_len, dim, dtype=dtype)

        attention = None
        if attn_implementation == "ring":
            attention = RingAttentionLlama(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dim=dim,
                max_seq_len=seq_len,
                ring_size=world_size,
                rotary_embed=True,
                use_striped=False,
            )
        elif attn_implementation == "striped_ring":
            attention = RingAttentionLlama(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dim=dim,
                max_seq_len=seq_len,
                ring_size=world_size,
                rotary_embed=True,
                use_striped=True,
            )
        elif attn_implementation == "eager":
            attention = Attention(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dim=dim,
                max_seq_len=seq_len,
                use_flash=False,
                rotary_embed=True,
            )
        elif attn_implementation == "flash":
            attention = Attention(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dim=dim,
                max_seq_len=seq_len,
                use_flash=True,
                rotary_embed=True,
            )
        else:
            raise ValueError(f"Unknown attention implementation: {attn_implementation}")

        attention = attention.to(dtype)
        input = input.cuda(rank)
        attention = attention.cuda(rank)

        for _ in range(10):
            _ = attention.forward_attention(input)
        torch.cuda.synchronize()

        if profile:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
                record_shapes=True,
                with_stack=True
            ) as prof:
                for i in range(6):
                    with torch.profiler.record_function("forward_attention"):
                        _ = attention.forward_attention(input)
                    prof.step()

            prof.export_chrome_trace(f"./logs/{attn_implementation}_rank{rank}_seq_len{seq_len}.json")
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
        
        else:
            times = []
            for _ in range(num_iters):
                torch.cuda.synchronize()
                start = time.time()
                _ = attention.forward_attention(input)
                torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)
            avg_time = sum(times) / len(times)
            print(f"Sequence Length: {seq_len}")
            print(f"{attn_implementation}: {avg_time * 1000:.3f} ms")
            return avg_time

    finally:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark attention variants")
    parser.add_argument(
        "--runtime",
        type=str,
        help="Which attention runtimes to benchmark. Options: ring, striped_ring, flash, eager",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiling. Defaults to False.",
    )
    args = parser.parse_args()
    print("Timing attention modules...")

    assert (
        args.world_size <= torch.cuda.device_count()
    ), f"world size {args.world_size} must be less than the number of cuda devices {torch.cuda.device_count()}"

    if "ring" not in args.runtime:
        assert args.world_size == 1

    #batch_size = 2
    #seq_len = 2048
    batch_size = args.batch_size
    seq_len = args.seq_len
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    num_iters = 100

    mp.spawn(
        benchmark_attention,
        args=(
            args.world_size,
            batch_size,
            seq_len,
            args.runtime,
            dim,
            n_heads,
            n_kv_heads,
            args.dtype,
            num_iters,
            args.profile
        ),
        nprocs=args.world_size,
        join=True,
    )
