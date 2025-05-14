import argparse
import json
import os
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from math import ceil
from contextlib import nullcontext

from ring_attention_pytorch.inference.distributed import (
    get_rank,
    get_world_size,
    setup,
    cleanup,
)
import torch
import utils
from torch.profiler import record_function
from transformers import AutoTokenizer

from ring_attention_pytorch.inference.args import ModelArgs
from ring_attention_pytorch.inference.cache import DecodingCache
from ring_attention_pytorch.inference.custom_model import Transformer


@dataclass
class SamplingArgs:
    temperature: float = 0.6
    max_output_tokens: int = 512


class LLM:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        params_file: str,
        device: torch.device | str,
    ) -> "LLM":
        """
        Load a Llama or Code Llama checkpoint and return a new
        generator for this model.
        """
        print("Start building model")
        start_time = time.time()

        ckpt_path = Path(ckpt_dir) / "consolidated.00.pth"
        with resources.files(
            "ring_attention_pytorch.inference.ring_llama_params"
        ).joinpath(params_file).open() as f:
            model_args = ModelArgs(**json.load(f))

        dtype = getattr(torch, model_args.dtype)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        torch.set_default_device(device)
        torch.set_default_dtype(dtype)

        model = Transformer(model_args)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        print(f"loaded model in {time.time() - start_time:.2f} seconds")

        return LLM(model_args, model, tokenizer, device, dtype)

    def __init__(
        self,
        model_args: ModelArgs,
        model: Transformer,
        tokenizer: AutoTokenizer,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.model_args = model_args
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

    def call_model(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
        cache: DecodingCache | None,
        cache_pos: torch.Tensor | None,
        iter_num: int,
        auto_shard_seq: bool,
        use_fast_ring_decoding: bool,
    ):
        if tokens.shape[1] > 1:
            with record_function("prefill"):
                logits = self.model.forward(
                    tokens=tokens,
                    attn_mask=mask,
                    input_pos=input_pos,
                    auto_shard_seq=auto_shard_seq,
                    cache=cache,
                    cache_pos=cache_pos,
                )
        else:
            with record_function("incremental_gen"):
                logits = self.model.forward(
                    tokens=tokens,
                    attn_mask=mask,
                    input_pos=input_pos,
                    auto_shard_seq=auto_shard_seq and cache is None,
                    cache=cache,
                    cache_pos=cache_pos,
                    use_fast_ring_decoding=use_fast_ring_decoding,
                )

        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[list[int]],
        sampling_args: SamplingArgs,
        use_cache: bool = True,
        use_fast_ring_decoding: bool = False,
    ) -> list[list[int]]:
        bsz = len(prompt_tokens)
        params = self.model.params
        auto_shard_seq = False
        if params.attn_implementation == "ring":
            auto_shard_seq = True

        padded_inputs, mask, input_pos = utils.collate(prompt_tokens, self.device)
        max_prompt_len = padded_inputs.size()[1]
        max_len = sampling_args.max_output_tokens + max_prompt_len
        max_len = ceil(max_len / get_world_size()) * get_world_size()
        assert max_len < params.max_seq_len

        out_tokens = torch.zeros(
            (bsz, sampling_args.max_output_tokens), dtype=torch.long
        )
        eos_reached = torch.tensor([False] * bsz)
        cache = None
        cache_pos = None
        if use_cache:
            cache = self.model.make_cache(
                bsz=bsz,
                max_seqlen=max_len,
                device=self.device,
            )
            cache_pos = torch.arange(
                max_prompt_len, device=self.device, dtype=torch.long
            ).repeat(bsz, 1)

        cur_tokens = padded_inputs
        for iter_num in range(sampling_args.max_output_tokens):
            logits = self.call_model(
                cur_tokens,
                mask,
                input_pos,
                cache,
                cache_pos,
                iter_num,
                auto_shard_seq,
                use_fast_ring_decoding,
            )

            next_token = utils.sample(logits, sampling_args.temperature)
            out_tokens[:, iter_num] = next_token.squeeze()
            eos_reached |= (next_token == self.tokenizer.eos_token_id).squeeze()
            if use_cache:
                # No need to shard the sequence during decoding
                auto_shard_seq = False
                cur_tokens = next_token
                mask = None
                input_pos = input_pos[:, -1:] + 1
                cache_pos = cache_pos[:, -1:] + 1
            else:
                cur_tokens = torch.cat((cur_tokens, next_token), dim=1)
                mask = torch.cat(
                    (mask, torch.ones((bsz, 1), dtype=torch.bool, device=mask.device)),
                    dim=1,
                )
                input_pos = torch.cat((input_pos, input_pos[:, -1:] + 1), dim=1)

            if all(eos_reached):
                break

        responses = []
        for tokens in out_tokens.tolist():
            if self.tokenizer.eos_token_id in tokens:
                responses.append(
                    tokens[: tokens.index(self.tokenizer.eos_token_id) + 1]
                )
            else:
                responses.append(tokens)
        return responses


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    params_file: str,
    use_cache: bool,
    use_fast_ring_decoding: bool,
    max_completion_tokens: int = 512,
    context_len: int = 32000,
    profile: bool = False,
):
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        world_size = 1
        local_rank = 0

    if use_fast_ring_decoding:
        assert use_cache

    device = setup(world_size, local_rank)
    warmup_iterations = 1

    llama = LLM.build(ckpt_dir, tokenizer_path, params_file, device)

    # with open("prompts.jsonl", "r") as f:
    #     prompts = [json.loads(line) for line in f]
    with open(f"prompt_{context_len}_tokens.jsonl", "r") as f:
        prompts = [[json.loads(line)] for line in f]

    prompt_tokens = [
        llama.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    sampling_args = SamplingArgs(temperature=0, max_output_tokens=max_completion_tokens)
    if profile:
        sampling_args.max_output_tokens = 5

        for i in range(warmup_iterations):
            print(f"Warmup iteration {i+1}")
            _ = llama.generate(
                prompt_tokens,
                sampling_args,
                use_cache=use_cache,
                use_fast_ring_decoding=use_fast_ring_decoding,
            )

    # Ensure GPU operations have completed
    torch.cuda.synchronize()

    # Benchmark Iterations
    print("Start main generation")
    start_time = time.perf_counter()
    total_tokens = 0
    with torch.profiler.profile(
        record_shapes=True,
    ) if profile else nullcontext() as prof:
        out_tokens = llama.generate(
            prompt_tokens,
            sampling_args,
            use_cache=use_cache,
            use_fast_ring_decoding=use_fast_ring_decoding,
        )
        total_tokens += sum(len(tokens) for tokens in out_tokens)

    # Ensure GPU operations have completed
    torch.cuda.synchronize()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    tokens_per_second = total_tokens / elapsed_time

    if prof:
        prof.export_chrome_trace(
            f"./logs/rank{local_rank}_generate_{llama.model_args.attn_implementation}_{'cache' if use_cache else 'no_cache'}_{'fast' if use_fast_ring_decoding else 'slow'}.json"
        )

    if get_rank() == 0:
        for i, prompt in enumerate(prompts):
            if len(prompt[0]['content']) < 200:
                print(f"> {prompt}")
            if len(out_tokens[i]) < 200:
                answer = llama.tokenizer.decode(out_tokens[i])
                print(answer)
                print("---------------")

        print(f"Total tokens: {total_tokens}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Tokens per second: {tokens_per_second:.2f} tokens/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Llama inference")
    parser.add_argument("ckpt_dir")
    parser.add_argument("tokenizer_path")
    parser.add_argument("params_file")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--use-fast-ring-decoding", action="store_true")
    parser.add_argument("--context-len",type=int,default=32000,)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    try:
        main(
            args.ckpt_dir,
            args.tokenizer_path,
            args.params_file,
            args.use_cache,
            args.use_fast_ring_decoding,
            args.max_completion_tokens,
            args.context_len,
            args.profile,
        )
    finally:
        cleanup()
