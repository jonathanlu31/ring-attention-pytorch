import os
from functools import cache

import torch
import torch.distributed as dist


def setup(rank, world_size, seed=42):
    assert rank < world_size

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(42)
    return device


@cache
def get_rank():
    return dist.get_rank()
