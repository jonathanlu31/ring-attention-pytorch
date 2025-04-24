import torch.nn as nn
import torch
from ring_attention_pytorch.ring_attention import RingRotaryEmbedding, maybe_pad_seq_and_mask, sharded_batch_to_sharded_seq, sharded_seq_to_sharded_batch, apply_rotary_pos_emb
from einops import rearrange
from ring_attention_pytorch.ring_flash_attention_cuda import ring_flash_attn_cuda

class RingAttentionLlama(nn.Module):
    def __init__(
        self,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        dim: int = 4096,
        bucket_size: int = 512,
        ring_seq_size: int = 512,
        use_striped: bool = False,
        auto_shard_seq: bool = False,
        rotary_embed: bool = False,
        rotary_embed_theta: int = 10000,
        ring_size: int = 1,
    ):
        super().__init__()
        # whether to use flash attention cuda kernel
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads

        assert dim % n_heads == 0
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.use_striped = use_striped

        self.auto_shard_seq = auto_shard_seq # this should be done at the transformer level on the token ids for efficiency, but for testing purposes
        self.ring_seq_size = ring_seq_size
        self.ring_size = ring_size
        self.bucket_size = bucket_size

        # TODO: rotary

        self.rotary_embed = None
        # if rotary_embed:
        #     self.rotary_embed = RingRotaryEmbedding(
        #         dim = self.head_dim,
        #         ring = True,
        #         striped = self.use_striped,
        #         theta = rotary_embed_theta,
        #         buckets = ring_seq_size // bucket_size
        #     )

        self.wq = nn.Linear(
            dim,
            self.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim,
            dim,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        einstein notation

        b - batch
        h - heads
        d - feature dimension
        n, i, j - sequence
        """
        bsz, seqlen, _ = x.shape
        striped_bucket_size = self.bucket_size if not self.use_striped else self.ring_seq_size


        if self.auto_shard_seq:
            x, mask = maybe_pad_seq_and_mask(x, mask, self.ring_seq_size)

            if self.striped_ring_attn:
                x = rearrange(x, 'b (i j) d -> b (j i) d', i = striped_bucket_size)

                if mask is not None:
                    mask = rearrange(mask, 'b (i j) -> b (j i)', i = striped_bucket_size)

            (x, mask), batch_sizes, num_sharded_batches = sharded_batch_to_sharded_seq(x, mask, self.ring_seq_size)
            
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # rotary relative positions

        if self.rotary_embed is not None:
            rotary_emb = self.rotary_embed(q.shape[-3])
            xq = apply_rotary_pos_emb(rotary_emb, xq)
            xk = apply_rotary_pos_emb(rotary_emb, xk)


        out = ring_flash_attn_cuda(
            xq, xk, xv,
            mask,
            causal=True,
            bucket_size=self.bucket_size,
            ring_reduce_col=self.ring_size > 1,
            striped_ring_attn=self.use_striped,
            ring_size=self.ring_size,
        )

        # combine heads

        out = rearrange(out, 'b n h d -> b n (h d)')
        out = self.wo(out)

        if self.auto_shard_seq:
            out = sharded_seq_to_sharded_batch(out, batch_sizes, num_sharded_batches)

            if self.striped_ring_attn:
                out = rearrange(out, 'b (j i) d -> b (i j) d', i = striped_bucket_size)

            out = out[:, :seqlen]

        return out
