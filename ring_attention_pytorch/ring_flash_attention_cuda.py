from __future__ import annotations

from math import ceil

import torch
from torch import nn, einsum, Tensor
from torch.autograd.function import Function
from torch.amp import autocast

from ring_attention_pytorch.ring import (
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank,
    get_world_size
)
from ring_attention_pytorch.inference.cache import DecodingCache

from beartype import beartype

from einops import rearrange, repeat, reduce

from ring_attention_pytorch.inference.distributed import all_gather

# helpers

def exists(v):
    return v is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

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

# ring + (flash) attention forwards and backwards

# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf
# ring attention - https://arxiv.org/abs/2310.01889

class RingFlashAttentionCUDAFunction(Function):

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
        causal: bool,
        # bucket_size: int,
        ring_reduce_col: bool,
        striped_ring_attn: bool,
        max_lookback_seq_len: int | None,
        ring_size: int | None,
        softclamp_qk_sim: bool,
        softclamp_value: float,
        layer_id: int | None,
        cache: DecodingCache | None,
        cache_pos: Tensor | None,
        use_fast_ring_decoding: bool,
    ):
        from ring_attention_pytorch.triton_flash_attn import flash_attn_forward

        assert k.shape[-2:] == v.shape[-2:]
        q_heads, kv_heads = q.shape[-2], k.shape[-2]

        assert divisible_by(q_heads, kv_heads)
        q_head_groups = q_heads // kv_heads

        assert all([t.is_cuda for t in (q, k, v)]), 'inputs must be all on cuda'

        dtype = q.dtype
        softmax_scale = q.shape[-1] ** -0.5

        if q.dtype == torch.float32:
            q = q.to(torch.bfloat16)

        if k.dtype == torch.float32:
            k = k.to(torch.bfloat16)

        if v.dtype == torch.float32:
            v = v.to(torch.bfloat16)

        ring_size = default(ring_size, get_world_size())

        cross_attn = q.shape[-3] != k.shape[-3]
        ring_reduce_col &= not cross_attn
        striped_ring_attn &= not cross_attn
        is_decoding = cache is not None and q.shape[1] == 1
        if is_decoding and use_fast_ring_decoding:
            ring_reduce_col = False

        assert k.shape[-1] == v.shape[-1], 'for simplicity when doing ring passing, assume dim_values is equal to dim_queries_keys, majority of transformer do this, not a big issue'

        # per_machine_seq_size = k.shape[-3]

        # calculate max ring passes

        max_ring_passes = None
        num_lookback_buckets = float('inf')

        # if exists(max_lookback_seq_len):
        #     assert causal
        #     assert not (ring_reduce_col and not divisible_by(per_machine_seq_size, bucket_size))

        #     max_ring_passes = ceil(max_lookback_seq_len / per_machine_seq_size)
        #     num_lookback_buckets = max_lookback_seq_len // bucket_size

        # ignore key padding mask if autoregressive

        if causal:
            mask = None

        # bucket_size = min(per_machine_seq_size, bucket_size)
        # per_machine_buckets = per_machine_seq_size // bucket_size

        orig_k, orig_v, orig_mask, q_seq_len, device = k, v, mask, q.shape[1], q.device

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        kv = torch.stack((k, v))

        # accumulated values

        # o - output
        # m - maximum
        # lse - logsumexp

        o = None
        m = None
        lse = None

        # non-causal and causal striped attention can have final normalization of output fused

        # can_fuse_final_output_normalization = not causal or (causal and striped_ring_attn) # TODO: Fix
        can_fuse_final_output_normalization = False

        # TODO: make this faster by sending all the tensors at once. Don't send each tensor in a separate send_receive_
        for (ring_rank, (is_first, is_last)), (kv, mask, cache_pos) in ring_pass_fn(kv, mask, cache_pos, max_iters = max_ring_passes, ring_size = ring_size):
            k, v = kv

            if cache:
                if q_seq_len == 1 and is_first:
                    # During decoding, you need to pull the cache on the first step
                    k, v = cache.update_and_get_kv(layer_id, k, v, cache_pos)
                elif q_seq_len > 1:
                    # During prefill, you just compute attention over the kv you receive but you need to update your cache
                    cache.update_kv(layer_id, k, v, cache_pos)

            # account for grouped query attention
            k, v = repeat_kv(k, q_head_groups), repeat_kv(v, q_head_groups)

            # translate key padding mask to bias

            bias = None

            if exists(mask):
                bias = torch.where(mask, 0.,  float('-inf'))

            # for non-striped attention
            # if the kv ring rank is equal to the current rank (block diagonal), then turn on causal
            # for striped attention, it is always causal, but a lt or gt sign needs to be changed to lte or gte within the cuda code, when determining masking out

            block_causal = False
            causal_mask_diagonal = False

            if causal:
                if striped_ring_attn:
                    block_causal = True
                    causal_mask_diagonal = get_rank() < ring_rank
                else:
                    block_causal = get_rank() == ring_rank

                    if get_rank() < ring_rank:
                        continue

            o, m, lse = flash_attn_forward(
                q, k, v,
                causal = block_causal,
                o = o,
                m = m,
                lse = lse,
                bias = bias,
                softmax_scale = softmax_scale,
                causal_mask_diagonal = causal_mask_diagonal,
                return_normalized_output = can_fuse_final_output_normalization and is_last,
                load_accumulated = not is_first,
                softclamp_qk_sim = softclamp_qk_sim,
                softclamp_value = softclamp_value
            )

        if use_fast_ring_decoding:
            assert not can_fuse_final_output_normalization
            # o should be shape [B, 1, H, D], m should be [B, H, 1], lse should be [B, H, 1]
            o, m, lse = o[:, :q_seq_len], m[..., :q_seq_len], lse[..., :q_seq_len]
            os, ms, lses = all_gather(o), all_gather(m), all_gather(lse)  # TODO: fix this to use only one all_gather
            os, ms, lses = torch.stack(os), torch.stack(ms), torch.stack(lses)

            m_max = torch.max(ms, dim=0, keepdim=True)[0]
            m_max_norm = torch.exp(ms - m_max)

            ses = torch.exp(lses - ms)
            ses_norm = torch.sum(ses * m_max_norm, dim=0)  # [B, H, 1]

            os = os * rearrange(m_max_norm, 'w b h n -> w b n h 1')
            o = torch.sum(os, dim=0) / rearrange(ses_norm, 'b h n -> b n h 1')
        else:
            if not can_fuse_final_output_normalization:
                m = m[..., :q_seq_len]

                o_scale = torch.exp(m - lse[..., :q_seq_len])
                o.mul_(rearrange(o_scale, 'b h n -> b n h 1'))

        ctx.args = (
            causal,
            softmax_scale,
            orig_mask,
            # bucket_size,
            ring_reduce_col,
            max_ring_passes,
            num_lookback_buckets,
            striped_ring_attn,
            ring_size,
            q_head_groups,
            softclamp_qk_sim,
            softclamp_value,
            dtype
        )

        ctx.save_for_backward(q, orig_k, orig_v, o, lse)

        # cast back to original dtype

        o = o.type(dtype)
        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):

        from ring_attention_pytorch.triton_flash_attn import flash_attn_backward

        (
            causal,
            softmax_scale,
            mask,
            bucket_size,
            ring_reduce_col,
            max_ring_passes,
            num_lookback_buckets,
            striped_ring_attn,
            ring_size,
            q_head_groups,
            softclamp_qk_sim,
            softclamp_value,
            dtype
        ) = ctx.args

        q, k, v, o, lse = ctx.saved_tensors

        do = do.type(o.dtype)

        device = q.device

        if causal:
            mask = None

        row_length = q.shape[-3]

        per_machine_seq_size = k.shape[-3]
        per_machine_buckets = per_machine_seq_size // bucket_size

        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass

        device = q.device

        dq = torch.zeros(q.shape, device = device, dtype = torch.float32)
        dk = torch.zeros_like(k, device = device)
        dv = torch.zeros_like(v, device = device)

        # k and v will have 16 bits, and dk and dv can also be accumulated safely with the same type, i think

        assert k.dtype == v.dtype
        kv_dtype = k.dtype

        kv_and_dkv = torch.stack((k, v, dk, dv))

        # receive buffers, to be alternated with sent buffer

        receive_kv_and_dkv = None
        receive_mask = None

        # caching the delta (do * o for backwards pass) across ring reduce

        delta = None

        for (ring_rank, _), ((kv_and_dkv, mask), (receive_kv_and_dkv, receive_mask)) in ring_pass_fn(kv_and_dkv, mask, receive_buffers = (receive_kv_and_dkv, receive_mask), max_iters = max_ring_passes, ring_size = ring_size):

            k, v, dk, dv = kv_and_dkv

            # account for grouped query attention

            k, v = map(lambda t: repeat(t, '... h d -> ... (g h) d', g = q_head_groups), (k, v))

            # translate key padding mask to bias

            bias = None

            if exists(mask):
                bias = torch.where(mask, 0., float('-inf'))
                bias = rearrange(bias, 'b j -> b 1 1 j')

            # determine whether to do causal mask or not
            # depends on whether it is striped attention, as well as current machine rank vs ring rank

            if causal and striped_ring_attn:
                need_accum = True
                block_causal = True
                causal_mask_diagonal = get_rank() < ring_rank
            elif causal:
                need_accum = get_rank() >= ring_rank
                block_causal = get_rank() == ring_rank
                causal_mask_diagonal = False
            else:
                need_accum = True
                block_causal = False
                causal_mask_diagonal = False

            # use flash attention backwards kernel to calculate dq, dk, dv and accumulate

            if need_accum:
                ring_dq = torch.empty(q.shape, device = device, dtype = torch.float32)
                ring_dk = torch.empty_like(k)
                ring_dv = torch.empty_like(v)

                with torch.inference_mode():
                    delta = flash_attn_backward(
                        do,
                        q,
                        k,
                        v,
                        o,
                        lse,
                        ring_dq,
                        ring_dk,
                        ring_dv,
                        delta = delta,
                        bias = bias,
                        causal = block_causal,
                        causal_mask_diagonal = causal_mask_diagonal,
                        softmax_scale = softmax_scale,
                        softclamp_qk_sim = softclamp_qk_sim,
                        softclamp_value = softclamp_value,
                    )

                # account for grouped query attention

                ring_dk = reduce(ring_dk, '... (g h) d -> ... h d', g = q_head_groups, reduction = 'sum')
                ring_dv = reduce(ring_dv, '... (g h) d -> ... h d', g = q_head_groups, reduction = 'sum')

                dq.add_(ring_dq)
                dk.add_(ring_dk)
                dv.add_(ring_dv)

            if not ring_reduce_col:
                continue

            dkv = kv_and_dkv[2:]

            max_ring_passes = default(max_ring_passes, ring_size)
            dkv = ring_pass(ring_size - max_ring_passes + 1, dkv)

            dk, dv = dkv

        dq, dk, dv = map(lambda t: t.to(dtype), (dq, dk, dv))

        return dq, dk, dv, None, None, None, None, None, None, None, None, None

ring_flash_attn_cuda_ = RingFlashAttentionCUDAFunction.apply

@autocast('cuda', enabled = False)
@beartype
def ring_flash_attn_cuda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    causal: bool = False,
    ring_reduce_col: bool = False,
    striped_ring_attn: bool = False,
    max_lookback_seq_len: int | None = None,
    ring_size: int | None = None,
    softclamp_qk_sim: bool = False,
    softclamp_value: float = 50.,
    layer_id: int | None = None,
    cache: DecodingCache | None = None,
    cache_pos: Tensor | None = None,
    use_fast_ring_decoding: bool = False
):
    return ring_flash_attn_cuda_(q, k, v, mask, causal, ring_reduce_col, striped_ring_attn, max_lookback_seq_len, ring_size, softclamp_qk_sim, softclamp_value, layer_id, cache, cache_pos, use_fast_ring_decoding)
