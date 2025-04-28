import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from distributed import all_gather, get_rank, is_distributed
from einops import rearrange
from math import ceil

from ring_attention_pytorch.ring_flash_attention_cuda import ring_flash_attn_cuda

##############
# Rotary Positional Embeddings
##############


def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * torch.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    return torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
        new_freqs,
    )


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1])
    shape = [d if i in [0, 1, ndim - 1] else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


##############
# Ring Attention
##############


def pad_to_multiple(
    x: torch.Tensor,
    length: int,
    pad_value: int = 0,
    padding_side: Literal["left", "right"] = "left",
):
    seq_len = x.shape[1]
    remainder = seq_len % length

    if remainder == 0:
        return x, 0

    pad_len = length - remainder
    padded_shape = list(x.shape)
    padded_shape[1] += pad_len

    t = torch.full(padded_shape, pad_value, dtype=x.dtype, device=x.device)

    if padding_side == "left":
        t[:, pad_len:] = x
    else:
        t[:, :seq_len] = x

    return t, pad_len


def maybe_pad_seq_and_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    pos: torch.Tensor,
    seq_size: int,
    pad_value: int = 0,
):
    bsz, seq_len = x.shape[:2]

    # auto pad sequence and mask, as ring passing makes assumption tensor is all same shape

    x, pad_length = pad_to_multiple(x, seq_size, pad_value=pad_value)
    pos, _ = pad_to_multiple(pos, seq_size, pad_value=-1)

    if pad_length == 0:
        return (x, mask, pos), pad_length

    if mask is None:
        mask = torch.ones((bsz, seq_len), device=x.device).bool()

    mask, _ = pad_to_multiple(mask, seq_size, pad_value=False)

    return (x, mask, pos), pad_length


def shard_seq(
    x: torch.Tensor, mask: torch.Tensor | None, pos: torch.Tensor, seq_size: int
):
    assert is_distributed()
    assert x.shape[1] % seq_size == 0

    rank = get_rank()
    x_split, pos_split = x.split(seq_size, dim=1), pos.split(seq_size, dim=1)
    x, pos = x_split[rank], pos_split[rank]

    if mask is not None:
        mask_split = mask.split(seq_size, dim=1)
        mask = mask_split[rank]

    return x, mask, pos


def join_seq(logits: torch.Tensor) -> torch.Tensor:
    logits_list = all_gather(logits)
    return torch.cat(logits_list, dim=1)


class RingAttentionLlama(nn.Module):
    def __init__(
        self,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        dim: int = 4096,
        ring_size: int = 1,
        use_striped: bool = False,
        rotary_embed: bool = False,
        max_seq_len: int = -1,
        rope_theta: int = 10000,
        use_scaled_rope: bool = False,
    ):
        super().__init__()
        # whether to use flash attention cuda kernel
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads

        assert dim % n_heads == 0
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.use_striped = use_striped
        self.ring_size = ring_size

        if rotary_embed:
            self.freqs_cis = precompute_freqs_cis(
                self.head_dim,
                max_seq_len * 2,
                rope_theta,
                use_scaled_rope,
            )

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
    ):
        """
        einstein notation

        b - batch
        h - heads
        d - feature dimension
        n, i, j - sequence
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # rotary relative positions
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        out = ring_flash_attn_cuda(
            xq,
            xk,
            xv,
            mask,
            causal=True,
            ring_reduce_col=self.ring_size > 1,
            striped_ring_attn=self.use_striped,
            ring_size=self.ring_size,
        )

        # combine heads
        out = rearrange(out, "b n h d -> b n (h d)")
        return self.wo(out)

    def forward_attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ):
        ring_seq_size = ceil(x.shape[1] / self.ring_size)
        if pos is None:
            pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device).repeat(x.shape[0], 1)

        # If batching, x, mask, and pos should already be padded to max seq len in the batch
        # This pads the sequence further to make it a multiple of ring_seq_size
        (x, mask, pos), pad_length = maybe_pad_seq_and_mask(
            x, mask, pos, ring_seq_size
        )

        if self.use_striped:
            x = rearrange(x, "b (i j) d -> b (j i) d", i=ring_seq_size)
            pos = rearrange(pos, "b (i j) -> b (j i)", i=ring_seq_size)

            if mask is not None:
                mask = rearrange(mask, "b (i j) -> b (j i)", i=ring_seq_size)

        x, mask, pos = shard_seq(x, mask, pos, ring_seq_size)
        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[pos]

        out = self(
            x=x,
            mask=mask,
            freqs_cis=freqs_cis,
        )

        out = join_seq(out)

        if self.use_striped:
            out = rearrange(out, "b (j i) d -> b (i j) d", i=ring_seq_size)

        return out[:, pad_length:]


##############
# Eager Attention
##############


class Attention(nn.Module):
    def __init__(
        self,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        dim: int = 4096,
        rotary_embed: bool = False,
        max_seq_len: int = -1,
        rope_theta: int = 10000,
        use_scaled_rope: bool = False,
    ):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        world_size = 1
        self.n_local_heads = n_heads // world_size
        self.n_local_kv_heads = self.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        if rotary_embed:
            self.freqs_cis = precompute_freqs_cis(
                self.head_dim,
                max_seq_len * 2,
                rope_theta,
                use_scaled_rope,
            )

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # self.cache_k = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )
        # self.cache_v = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        # TODO: Fix
        # repeat k/v heads if n_kv_heads < n_heads
        xk = repeat_kv(
            xk, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        xv = repeat_kv(
            xv, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def forward_attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ):
        if pos is None:
            pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device).repeat(x.shape[0], 1)

        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[pos]

        seqlen = x.shape[1]
        if seqlen > 1 and mask is None:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)

        out = self(x=x, mask=mask, freqs_cis=freqs_cis, start_pos=0)

        return out
