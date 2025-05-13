from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.amp import autocast

from ring_attention_pytorch.ring import (
    ring_pass,
    all_ring_pass,
    null_ring_pass,
    one_ring_pass,
    get_rank,
    get_world_size,
)
from ring_attention_pytorch.inference.cache import DecodingCache
from flash_attn.flash_attn_interface import (
    _flash_attn_forward,
    _flash_attn_varlen_forward,
)

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


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out is None or (torch.all(out == 0) and torch.all(torch.isinf(lse))):
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif torch.all(block_out == 0) and torch.all(torch.isinf(block_lse)):
        # If there's an empty block, corresponding to an empty KV cache on a particular device, it should
        # have no impact on the output
        return out, lse
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


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
        # q shape: [B, q_seq_len, H, D]
        # k shape: [B, kv_seq_len, H, D]
        # v shape: [B, kv_seq_len, H, D]
        from ring_attention_pytorch.triton_flash_attn import flash_attn_forward

        assert k.shape[-2:] == v.shape[-2:]
        q_heads, kv_heads = q.shape[-2], k.shape[-2]

        assert divisible_by(q_heads, kv_heads)
        q_head_groups = q_heads // kv_heads

        assert all([t.is_cuda for t in (q, k, v)]), "inputs must be all on cuda"

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
        assert not (is_decoding and causal), "Decoding doesn't use causal"

        if is_decoding and use_fast_ring_decoding:
            ring_reduce_col = False

        assert k.shape[-1] == v.shape[-1], (
            "for simplicity when doing ring passing, assume dim_values is equal to dim_queries_keys, majority of transformer do this, not a big issue"
        )

        # per_machine_seq_size = k.shape[-3]

        # calculate max ring passes

        max_ring_passes = None
        num_lookback_buckets = float("inf")

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

        cache_mask = None
        if is_decoding:
            k, v, cache_mask = cache.update_and_get_kv(layer_id, k, v, cache_pos)

        kv = torch.stack((k, v))

        # accumulated values

        # o - output
        # m - maximum
        # lse - logsumexp

        o = None
        lse = None

        # non-causal and causal striped attention can have final normalization of output fused

        # TODO: make this faster by sending all the tensors at once. Don't send each tensor in a separate send_receive_
        for (ring_rank, (is_first, is_last)), (
            kv,
            mask,
            cache_pos,
            cache_mask,
        ) in ring_pass_fn(
            kv,
            mask,
            cache_pos,
            cache_mask,
            max_iters=max_ring_passes,
            ring_size=ring_size,
        ):
            k, v = kv
            if cache_mask is not None:
                k, v = k[:, cache_mask], v[:, cache_mask]

            if cache and q_seq_len > 1:
                # During prefill, you just compute attention over the kv you receive but you need to update your cache
                cache.update_kv(layer_id, k, v, cache_pos)

            # account for grouped query attention
            k, v = repeat_kv(k, q_head_groups), repeat_kv(v, q_head_groups)

            # for non-striped attention
            # if the kv ring rank is equal to the current rank (block diagonal), then turn on causal
            # for striped attention, it is always causal, but a lt or gt sign needs to be changed to lte or gte within the cuda code, when determining masking out

            use_causal = False
            slice_ = None
            q_input = q
            k_input = k
            v_input = v

            if causal:
                if striped_ring_attn:
                    use_causal = True
                    if get_rank() < ring_rank:
                        q_input = q_input[:, 1:]
                        k_input = k_input[:, :-1]
                        v_input = v_input[:, :-1]
                        slice_ = (slice(None), slice(1, None))
                else:
                    use_causal = get_rank() == ring_rank

                    if get_rank() < ring_rank:
                        continue

            outputs = _flash_attn_forward(
                q=q_input,
                k=k_input,
                v=v_input,
                causal=use_causal,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
            # block_lse [B, H, seq_len]

            o, lse = update_out_and_lse(o, lse, block_out, block_lse, slice_)

        if use_fast_ring_decoding:
            # o is shape [B, 1, H, D], lse is shape [B, 1, H, 1]
            B, _, H, D = o.shape
            block_o, block_lse = o[:, :q_seq_len], lse[:, :q_seq_len]
            o_flat, lse_flat = (
                block_o.view(o.shape[0], -1),
                block_lse.view(lse.shape[0], -1),
            )
            combined = torch.cat((o_flat, lse_flat), dim=-1)  # [B, total_feature_dim]
            gathered = all_gather(combined)  # [world_size, B, total_feature_dim]
            gathered = torch.stack(gathered)

            os_flat, lses_flat = gathered[:, :, :o_flat.shape[1]], gathered[:, :, o_flat.shape[1]:]
            os, lses = os_flat.view(-1, B, 1, H, D), lses_flat.view(-1, B, 1, H, 1)

            # Replace +inf with -inf in lses
            lses = torch.where(torch.isinf(lses) & (lses > 0), torch.full_like(lses, float('-inf')), lses)

            new_lse = torch.logsumexp(lses, dim=0, keepdim=True)
            renormalized_os = torch.exp(lses - new_lse) * os
            o = torch.sum(renormalized_os, dim=0)

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
            dtype,
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
            dtype,
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

        dq = torch.zeros(q.shape, device=device, dtype=torch.float32)
        dk = torch.zeros_like(k, device=device)
        dv = torch.zeros_like(v, device=device)

        # k and v will have 16 bits, and dk and dv can also be accumulated safely with the same type, i think

        assert k.dtype == v.dtype
        kv_dtype = k.dtype

        kv_and_dkv = torch.stack((k, v, dk, dv))

        # receive buffers, to be alternated with sent buffer

        receive_kv_and_dkv = None
        receive_mask = None

        # caching the delta (do * o for backwards pass) across ring reduce

        delta = None

        for (ring_rank, _), (
            (kv_and_dkv, mask),
            (receive_kv_and_dkv, receive_mask),
        ) in ring_pass_fn(
            kv_and_dkv,
            mask,
            receive_buffers=(receive_kv_and_dkv, receive_mask),
            max_iters=max_ring_passes,
            ring_size=ring_size,
        ):
            k, v, dk, dv = kv_and_dkv

            # account for grouped query attention

            k, v = map(
                lambda t: repeat(t, "... h d -> ... (g h) d", g=q_head_groups), (k, v)
            )

            # translate key padding mask to bias

            bias = None

            if exists(mask):
                bias = torch.where(mask, 0.0, float("-inf"))
                bias = rearrange(bias, "b j -> b 1 1 j")

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
                ring_dq = torch.empty(q.shape, device=device, dtype=torch.float32)
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
                        delta=delta,
                        bias=bias,
                        causal=block_causal,
                        causal_mask_diagonal=causal_mask_diagonal,
                        softmax_scale=softmax_scale,
                        softclamp_qk_sim=softclamp_qk_sim,
                        softclamp_value=softclamp_value,
                    )

                # account for grouped query attention

                ring_dk = reduce(
                    ring_dk, "... (g h) d -> ... h d", g=q_head_groups, reduction="sum"
                )
                ring_dv = reduce(
                    ring_dv, "... (g h) d -> ... h d", g=q_head_groups, reduction="sum"
                )

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


@autocast("cuda", enabled=False)
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
    softclamp_value: float = 50.0,
    layer_id: int | None = None,
    cache: DecodingCache | None = None,
    cache_pos: Tensor | None = None,
    use_fast_ring_decoding: bool = False,
):
    return ring_flash_attn_cuda_(
        q,
        k,
        v,
        mask,
        causal,
        ring_reduce_col,
        striped_ring_attn,
        max_lookback_seq_len,
        ring_size,
        softclamp_qk_sim,
        softclamp_value,
        layer_id,
        cache,
        cache_pos,
        use_fast_ring_decoding,
    )
