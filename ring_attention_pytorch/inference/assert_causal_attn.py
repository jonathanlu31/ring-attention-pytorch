import os
from math import ceil

import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from attention_variants import Attention, RingAttentionLlama


def setup(
    rank,
    world_size,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    backend = "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def start(
    rank,
    world_size,
    batch_size,
    batch_size_var_len,
    seq_len,
    use_striped,
    dim,
    n_heads,
    n_kv_heads,
):
    setup(rank, world_size)
    print(f"rank {rank} started")

    ring_seq_size = ceil(seq_len / world_size)

    ring_attention = RingAttentionLlama(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        use_striped=use_striped,
        ring_seq_size=ring_seq_size,
        max_seq_len=seq_len,
        rotary_embed=True,
    ).to(torch.bfloat16)

    eager_attention = Attention(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=seq_len,
        rotary_embed=True,
    ).to(torch.bfloat16)

    eager_attention.load_state_dict(ring_attention.state_dict())

    # if batch_size_var_len:
    #     batch_size = batch_size + rank

    seq = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    # move to cuda if needed

    seq = seq.cuda(rank)
    eager_attention.cuda(rank)
    ring_attention.cuda(rank)

    # eager
    with torch.inference_mode():
        eager_out = eager_attention.forward_attention(seq)

    # ring
    with torch.inference_mode():
        ring_out = ring_attention.forward_attention(seq)

    # validate output is the same for sequence split across machines vs without

    if rank == 0:
        ring_attention = ring_attention.cpu()
        eager_attention = eager_attention.cpu()
        ring_out = ring_out.cpu()
        eager_out = eager_out.cpu()

        output_atol = 1e-2

        assert torch.allclose(
            ring_out, eager_out, atol=output_atol
        ), "output is not the same"
        print("✅ outputs are same between ring attention and eager attention")

    cleanup()


@click.command()
@click.option("--world-size", default=1, help="number of machines / processes")
@click.option("--batch-size", default=1, help="test batch size")
@click.option(
    "--batch-size-var-len", is_flag=True, help="test variable lengthed batch sizes"
)
@click.option(
    "--use-striped",
    is_flag=True,
    help="test striped ring attention from MIT follow up paper",
)
@click.option("--seq-len", default=3072, help="sequence length to test")
@click.option("--model-dim", default=1024, help="model dimensions for testing")
@click.option("--n-heads", default=8, help="number of query attention heads")
@click.option("--n-kv-heads", default=8, help="number of query attention head groups")
def test(
    world_size: int,
    batch_size: int,
    batch_size_var_len: bool,
    use_striped: bool,
    seq_len: int,
    model_dim: int,
    n_heads: int,
    n_kv_heads: int,
):
    assert (
        world_size <= torch.cuda.device_count()
    ), f"world size {world_size} must be less than the number of cuda devices {torch.cuda.device_count()}"

    mp.spawn(
        start,
        args=(
            world_size,
            batch_size,
            batch_size_var_len,
            seq_len,
            use_striped,
            model_dim,
            n_heads,
            n_kv_heads,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    test()
