"""vLLM offline inference client.

Uses vLLM's in-process `LLM` API (no HTTP server, no subprocess) to mirror
the paper's eval pipeline with one important simplification: the original
spawned a vLLM HTTP server and talked to it from many threads to saturate
the engine queue. The offline `LLM.generate(...)` call already does that
internally — it dispatches the entire prompt batch into vLLM's continuous
batcher in a single call.

Configuration parity with `paper/scripts/_vllm_server.py:VLLMServer`:
  dtype                      = bfloat16
  max_model_len              = 8192
  gpu_memory_utilization     = 0.85
  enforce_eager              = False
  max_num_seqs               = 256
  enable_chunked_prefill     = True
  trust_remote_code          = True

Generation parity with `paper/eval/client.py:VLLMClient`:
  temperature = 0    (greedy)
  no max_tokens cap  (vLLM clamps to max_model_len - prompt_len server-side)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

from dghard.inference.base import InferenceClient

logger = logging.getLogger(__name__)


class VLLMClient(InferenceClient):
    def __init__(
        self,
        ckpt: Union[str, Path],
        *,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = False,
        max_num_seqs: int = 256,
        enable_chunked_prefill: bool = True,
        trust_remote_code: bool = True,
        show_progress: bool = False,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as e:
            raise RuntimeError(
                "vLLM is not installed. Install with `pip install dghard[gpu]` "
                "or pass `--inference hf` to fall back to transformers.generate."
            ) from e
        self.ckpt = str(ckpt)
        self.max_model_len = int(max_model_len)
        logger.info(
            "VLLMClient: loading %s (dtype=%s max_model_len=%d gpu_util=%.2f "
            "max_num_seqs=%d enforce_eager=%s)",
            self.ckpt, dtype, max_model_len, gpu_memory_utilization,
            max_num_seqs, enforce_eager,
        )
        self.llm = LLM(
            model=self.ckpt,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            enable_chunked_prefill=enable_chunked_prefill,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.show_progress = bool(show_progress)

    # ---- prompt formatting --------------------------------------------

    def _format_prompt(self, prompt: str, system: Optional[str]) -> str:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            sys_part = f"{system}\n\n" if system else ""
            return f"{sys_part}{prompt}\n"

    # ---- generation ----------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        return self.chat_batch(
            [prompt], system=system, max_tokens=max_tokens, stop=stop,
        )[0]

    def chat_batch(
        self,
        prompts: Sequence[str],
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        progress_label: Optional[str] = None,
    ) -> list[str]:
        from vllm import SamplingParams

        if not prompts:
            return []
        formatted = [self._format_prompt(p, system) for p in prompts]

        # Per the no-cap policy: leave max_tokens None and vLLM uses
        # `max_model_len - prompt_len` server-side.
        sp_kwargs = {"temperature": 0.0, "top_p": 1.0}
        if max_tokens is not None:
            sp_kwargs["max_tokens"] = int(max_tokens)
        if stop:
            sp_kwargs["stop"] = list(stop)
        sp = SamplingParams(**sp_kwargs)

        # vLLM's offline engine handles continuous batching internally.
        outputs = self.llm.generate(formatted, sp, use_tqdm=self.show_progress)
        # `outputs` is in the same order as input prompts.
        return [o.outputs[0].text for o in outputs]
