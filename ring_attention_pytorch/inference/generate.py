import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import distributed
import torch
import utils
from args import ModelArgs
from model import Transformer
from stats import Stats
from torch.profiler import record_function
from transformers import AutoTokenizer


@dataclass
class SamplingArgs:
    temperature: float = 0.6
    max_output_tokens: int = 512


class LLM:
    GRAPH_WARMUPS: int = 3

    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        device: torch.device | str,
    ) -> "LLM":
        """
        Load a Llama or Code Llama checkpoint and return a new
        generator for this model.
        """
        start_time = time.time()

        ckpt_path = Path(ckpt_dir) / "consolidated.00.pth"
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(**params)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

        torch.set_default_device(device)
        torch.set_default_dtype(torch.bfloat16)

        model = Transformer(model_args)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        print(f"loaded model in {time.time() - start_time:.2f} seconds")

        return LLM(model_args, model, tokenizer, device)

    def __init__(
        self,
        model_args: ModelArgs,
        model: Transformer,
        tokenizer: AutoTokenizer,
        device: torch.device | str,
    ):
        self.model_args = model_args
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _compile_model(
        self,
        tokens_sliced: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
        cache,
    ):
        self._compiled_inputs = tuple(
            v.clone() for v in (tokens_sliced, mask, input_pos)
        )

        original_cache = ...
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):  # warmup runs
                _ = self.model.forward(*self._compiled_inputs)
        torch.cuda.current_stream().wait_stream(s)
        # TODO: reset cache to original cache

        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._compiled_logits = self.model.forward(*self._compiled_inputs)

        def replay(tokens, mask, input_pos: torch.Tensor, _cache):
            # cache is not updated here because it is updated in-place in the model
            self._compiled_inputs[0].copy_(tokens)
            self._compiled_inputs[1].copy_(mask)
            self._compiled_inputs[2].copy_(input_pos)

            self._cuda_graph.replay()

            return self._compiled_logits

        return replay

    def compile_and_call_model(
        self,
        tokens: torch.Tensor,
        mask,
        input_pos,
        cache,
        niter: int,
        use_cuda_graph: bool,
    ):
        if niter == 0:
            with record_function("prefill"):
                logits = self.model.forward(
                    tokens=tokens, mask=mask, input_pos=input_pos, cache=cache
                )
        else:
            with record_function("incremental_gen"):
                if self.compiled_model is None:
                    if use_cuda_graph:
                        self.compiled_model = self._compile_model(
                            tokens, mask, input_pos, cache=cache
                        )
                    else:
                        self.compiled_model = self.model.forward

                logits = self.compiled_model(
                    tokens=tokens, mask=mask, input_pos=input_pos, cache=cache
                )

        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[list[int]],
        sampling_args: SamplingArgs,
        use_cuda_graph: bool = True,
    ) -> tuple[Stats, list[list[int]]]:
        bsz = len(prompt_tokens)
        params = self.model.params

        padded_inputs, mask, _ = utils.collate(prompt_tokens, self.device)
        max_prompt_len = padded_inputs.size()[1]
        assert sampling_args.max_output_tokens + max_prompt_len < params.max_seq_length

        out_tokens = torch.zeros(
            (bsz, sampling_args.max_output_tokens), dtype=torch.long
        )
        eos_reached = torch.tensor([False] * bsz)
        cache = ...
        # cache = self.model.init_cache(
        #     args=self.model_args,
        #     length=bsz * max_seq_length,
        # )

        stats = Stats()
        stats.phase("total")
        cur_tokens = padded_inputs
        for niter in range(sampling_args.max_output_tokens):
            logits = self.compile_and_call_model(
                cur_tokens, cache, niter, use_cuda_graph
            )

            next_token = utils.sample(logits, sampling_args.temperature)

            next_token = next_token.reshape(-1)

            out_tokens[:, niter] = next_token
            eos_reached |= next_token == self.tokenizer.eos_id
            cur_tokens = next_token
            if all(eos_reached):
                break

        stats.end_phase(tokens=(niter + 1) * bsz)

        responses = []
        for tokens in out_tokens.tolist():
            if self.tokenizer.eos_id in tokens:
                responses.append(tokens[: tokens.index(self.tokenizer.eos_id) + 1])
            else:
                responses.append(tokens)
        return stats, responses


def main(ckpt_dir: str, tokenizer_path: str):
    if "WORLD_SIZE" in os.environ:
        mp_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        mp_size = 1
        local_rank = 0

    device = distributed.initialize(mp_size, local_rank)

    llama = LLM.build(ckpt_dir, tokenizer_path, device)

    with open("prompts.jsonl", "r") as f:
        prompts = [json.loads(line) for line in f]

    prompt_tokens = [
        llama.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        )
        for prompt in prompts
    ]
    sampling_args = SamplingArgs(temperature=0.6)
    stats, out_tokens = llama.generate(
        prompt_tokens, sampling_args, use_cuda_graph="NO_CUDA_GRAPHS" not in os.environ
    )

    if distributed.get_rank() == 0:
        for i, prompt in enumerate(prompts):
            print(f"> {prompt}")
            answer = llama.tokenizer.decode(out_tokens[i])
            print(answer)
            print("---------------")

        for phase_stats in stats.phases:
            print(phase_stats.show())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Llama inference")
    parser.add_argument("ckpt_dir")
    parser.add_argument("tokenizer_path")

    args = parser.parse_args()
    main(args.ckpt_dir, args.tokenizer_path)
