import torch

from ring_attention_pytorch.inference.distributed import get_rank, get_world_size


class DecodingCache:
    """
    A simple implementation of a cache for past values while decoding:
    - key/value vectors (kv cache)
    - role IDs
    - attention masks

    A DecodingCache is created for each (batched) generation and passed to the model in
    every forward pass. Each layer reads/writes to the same object independently.

    KV vectors are read/written *after* positional embeddings are applied.
    """

    def __init__(
        self,
        n_layers: int,
        bsz: int,
        max_seqlen: int,
        n_kv_heads: int,
        head_dim: int,
        use_ring: bool,
        use_striped: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.local_len = max_seqlen
        if use_ring:
            self.local_len = max_seqlen / get_world_size()
            if use_striped:
                raise NotImplementedError

        cache_shape = (n_layers, bsz, self.local_len, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.rank = get_rank()
        self.max_seen_cache_pos = 0

    @torch.no_grad()
    def update_kv(
        self,
        layer_id: int,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        cache_pos: torch.Tensor,
    ):
        """Used externally during ring attention prefill. Used internally by update_and_get_kv"""
        # cache_pos is global cache position, but only update the local cache if it falls within the local cache
        local_cache_indices = cache_pos - (self.rank * self.local_len)
        valid_local_idx_mask = (local_cache_indices >= 0) & (
            local_cache_indices < self.local_len
        )

        self.k_cache[layer_id, :, local_cache_indices[valid_local_idx_mask], :, :] = (
            k_val[valid_local_idx_mask]
        )
        self.v_cache[layer_id, :, local_cache_indices[valid_local_idx_mask], :, :] = (
            v_val[valid_local_idx_mask]
        )

        valid_indices = local_cache_indices[valid_local_idx_mask]
        self.max_seen_cache_pos = max(
            self.max_seen_cache_pos,
            (valid_indices.max() if valid_indices.numel() > 0 else 0),
        )

    @torch.no_grad()
    def update_and_get_kv(
        self,
        layer_id: int,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        cache_pos: torch.Tensor,
    ):
        """Used externally during normal attention. Used externally during ring attention decoding.
        """
        # k_val, v_val: [B, S, H, D]

        self.update_kv(layer_id, k_val, v_val, cache_pos)
        k_full = self.k_cache[layer_id, :, : self.max_seen_cache_pos + 1]
        v_full = self.v_cache[layer_id, :, : self.max_seen_cache_pos + 1]

        return k_full, v_full
