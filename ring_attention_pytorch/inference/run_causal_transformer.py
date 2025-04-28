# run_single_model.py
"""
Run one Transformer, save its output.

Example:
python run_single_model.py \
    --param-file ring_attention_pytorch/inference/ring_llama_params/3.1_8B.json \
    --output-file /tmp/llama_out.pt \
    --world-size 2 --batch-size 4 --seq-len 3072
"""

import json
import os
from importlib import resources

import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from args import ModelArgs
from custom_model import Transformer

from ring_attention_pytorch.inference.utils import collate


# -----------------------------------------------------------------------------#
# Distributed helpers
# -----------------------------------------------------------------------------#
def setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")


def cleanup() -> None:
    dist.destroy_process_group()


# -----------------------------------------------------------------------------#
# Worker
# -----------------------------------------------------------------------------#
def start(
    rank: int,
    world_size: int,
    param_file: str,
    output_file: str,
    batch_size: int,
    seq_len: int,
) -> None:
    setup(rank, world_size)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    with resources.files("ring_attention_pytorch.inference.ring_llama_params").joinpath(
        param_file
    ).open() as f:
        model_args = ModelArgs(**json.load(f))

    model = Transformer(model_args)

    dummy = [
        torch.randint(0, model_args.vocab_size, (seq_len - i,), dtype=torch.long)
        for i in range(batch_size)
    ]
    input_ids, attn_mask, pos = collate(dummy, device=f"cuda:{rank}")

    with torch.inference_mode():
        output = model.forward(
            input_ids,
            attn_mask,
            pos,
            auto_shard_seq=(model_args.attn_implementation == "ring"),
        )

    if rank == 0:
        torch.save(
            {
                "param_file": str(param_file),
                "world_size": world_size,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "output": output.cpu(),
            },
            output_file,
        )
        print(f"✅ Saved output tensor to {output_file}")

    cleanup()


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
@click.command()
@click.option("--param-file", required=True, help="Path to model-parameter JSON.")
@click.option("--output-file", required=True, help="Where to save model output (.pt).")
@click.option("--world-size", default=1, help="Distributed world size / GPU count.")
@click.option("--batch-size", default=1, help="Batch size (per rank).")
@click.option("--seq-len", default=3072, help="Sequence length.")
def main(
    param_file: str,
    output_file: str,
    world_size: int,
    batch_size: int,
    seq_len: int,
) -> None:
    assert world_size <= torch.cuda.device_count(), (
        f"--world-size {world_size} exceeds visible CUDA devices "
        f"({torch.cuda.device_count()})"
    )
    mp.spawn(
        start,
        args=(
            world_size,
            param_file,
            output_file,
            batch_size,
            seq_len,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
