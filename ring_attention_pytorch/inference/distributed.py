import os
from functools import cache

import torch
import torch.distributed as dist


def setup(world_size, rank, seed=42):
    assert rank < world_size

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(42)
    return device


@cache
def is_distributed():
    return dist.is_initialized()


@cache
def get_rank():
    return dist.get_rank()


@cache
def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def all_gather(t: torch.Tensor) -> list[torch.Tensor]:
    t = t.contiguous()
    world_size = get_world_size()
    gathered_tensors = [
        torch.empty_like(t, device=t.device, dtype=t.dtype) for i in range(world_size)
    ]
    dist.all_gather(gathered_tensors, t)
    return gathered_tensors

def cleanup():
    dist.destroy_process_group()