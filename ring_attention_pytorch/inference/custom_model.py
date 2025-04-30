# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import argparse
import json
from importlib import resources
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F
from args import ModelArgs
from distributed import get_world_size
from einops import rearrange
from torch import nn

from ring_attention_pytorch.inference.attention_variants import (
    Attention,
    RingAttentionLlama,
    join_seq,
    maybe_pad_seq_and_mask,
    precompute_freqs_cis,
    shard_seq,
)
from ring_attention_pytorch.inference.cache import DecodingCache


class RMSNorm(torch.nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, dtype=dtype)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, dtype: torch.dtype):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        assert not args.use_striped or args.attn_implementation == "ring"
        if args.attn_implementation == "ring":
            self.attention = RingAttentionLlama(
                layer_id=layer_id,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                dim=args.dim,
                use_striped=args.use_striped,
                ring_size=get_world_size(),
                dtype=dtype,
            )
        elif args.attn_implementation == "flash":
            self.attention = Attention(
                layer_id=layer_id,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                dim=args.dim,
                use_flash=True,
                dtype=dtype,
            )
        elif args.attn_implementation == "eager":
            self.attention = Attention(
                layer_id=layer_id,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                dim=args.dim,
                use_flash=False,
                dtype=dtype,
            )
        else:
            raise ValueError(
                f"Invalid attention implementation {args.attn_implementation}"
            )

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            dtype=dtype,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
        cache: DecodingCache | None = None,
        cache_pos: torch.Tensor | None = None,
        use_fast_ring_decoding: bool = False,
    ):
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask,
            cache=cache,
            cache_pos=cache_pos,
            use_fast_ring_decoding=use_fast_ring_decoding,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.dtype = torch.bfloat16
        if params.dtype is not None:
            self.dtype = getattr(torch, params.dtype)

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim, dtype=self.dtype
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, self.dtype))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps, dtype=self.dtype)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False, dtype=self.dtype
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        cache: DecodingCache | None = None,
        cache_pos: torch.Tensor | None = None,
        auto_shard_seq: bool = False,
        use_fast_ring_decoding: bool = False,
    ):
        _bsz, seqlen = tokens.shape

        if auto_shard_seq:
            assert self.params.attn_implementation == "ring"
            ring_seq_size = ceil(seqlen / get_world_size())

            # If batching, x, mask, and pos should already be padded to max seq len in the batch
            # This pads the sequence further to make it a multiple of ring_seq_size
            # FIXME: technically, this masking doesn't do anything because causal and mask don't play well in this codebase.
            (tokens, attn_mask, input_pos, cache_pos), pad_length = (
                maybe_pad_seq_and_mask(
                    x=tokens,
                    mask=attn_mask,
                    pos=input_pos,
                    cache_pos=cache_pos,
                    seq_size=ring_seq_size,
                )
            )

            if self.params.use_striped:
                tokens = rearrange(tokens, "b (i j) d -> b (j i) d", i=ring_seq_size)
                input_pos = rearrange(input_pos, "b (i j) -> b (j i)", i=ring_seq_size)

                if attn_mask is not None:
                    attn_mask = rearrange(
                        attn_mask, "b (i j) -> b (j i)", i=ring_seq_size
                    )

                if cache_pos is not None:
                    cache_pos = rearrange(
                        cache_pos, "b (i j) -> b (j i)", i=ring_seq_size
                    )

            tokens, attn_mask, input_pos, cache_pos = shard_seq(
                x=tokens,
                mask=attn_mask,
                pos=input_pos,
                cache_pos=cache_pos,
                seq_size=ring_seq_size,
            )

        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[input_pos]

        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            h = layer(
                h, freqs_cis=freqs_cis, mask=attn_mask, cache=cache, cache_pos=cache_pos, use_fast_ring_decoding=use_fast_ring_decoding
            )

        h = self.norm(h)
        output = self.output(h).float()

        if auto_shard_seq:
            output = join_seq(output)

            if self.params.use_striped:
                output = rearrange(output, "b (j i) d -> b (i j) d", i=ring_seq_size)

            return output[:, pad_length:]
        return output

    def make_cache(
        self, bsz: int, max_seqlen: int, device: torch.device
    ) -> DecodingCache:
        return DecodingCache(
            n_layers=self.n_layers,
            bsz=bsz,
            max_seqlen=max_seqlen,
            n_kv_heads=self.params.n_kv_heads,
            head_dim=self.params.dim // self.params.n_heads,
            device=device,
            dtype=self.dtype,
            use_ring=self.params.attn_implementation == "ring",
            use_striped=self.params.use_striped,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Llama model testing")
    parser.add_argument("ckpt_dir")
    parser.add_argument("params_name")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir  # Example: /global/homes/f/fogel/.llama/checkpoints/Llama3.1-8B/consolidated.00.pth

    with resources.files("ring_attention_pytorch.inference.ring_llama_params").joinpath(
        args.params_name
    ).open() as f:
        params = json.loads(f.read())

    args = ModelArgs(**params)
    model = Transformer(args)
    dtype = getattr(torch, args.dtype)
    model = model.to(dtype=dtype, device="cuda")
    model.eval()

    state_dict = torch.load(Path(ckpt_dir) / "consolidated.00.pth", map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
