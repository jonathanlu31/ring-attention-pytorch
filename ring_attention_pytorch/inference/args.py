from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048
    dtype: str = "bfloat16"

    # ring attention params
    use_striped: bool = True

    # # vision model params
    # vision_chunk_size: int = -1  # image resolution for image models
    # vision_max_num_chunks: int = 4
    # vision_num_cross_attention_layers: int = -1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0


# class LlamaModelArgs:
#     def __init__(self):
#         self.dim = 4096  # Hidden size
#         self.n_heads = 32
#         self.n_kv_heads = 8
#         self.n_layers = 32
#         self.vocab_size = 128256  # LLaMA 3 uses a 128K tokenizer
#         self.norm_eps = 1e-5
#         self.max_seq_len = 8192  # LLaMA 3 uses 8K context by default
#         self.max_batch_size = 1  # Change if you're using larger batches
#         self.rope_theta = 500000.0  # LLaMA 3 increased RoPE base
#         self.use_scaled_rope = True
#         self.multiple_of = 256  # For ffn hidden dim padding
        # self.ffn_dim_multiplier = 1.3  # Approx for LLaMA 3 8B

        # Optional: if you need dropout or other params
        # self.dropout = 0.0
