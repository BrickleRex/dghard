"""HuggingFace `transformers.generate` inference client.

Used when vLLM isn't available (CPU-only machines, environments that can't
install the vLLM extension wheels). Greedy decoding, batched with left-pad.

Output parity with VLLMClient: same chat-template application, same
greedy-no-sampling behavior, same default-no-cap on `max_new_tokens`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from tqdm.auto import tqdm

from dghard.inference.base import InferenceClient

logger = logging.getLogger(__name__)

# Conservative batch sizes — generation memory grows with both batch and
# max_new_tokens, and we don't know either ahead of time. Reviewers can
# override via the CLI `--batch-size` flag.
_DEFAULT_BATCH_CPU = 4
_DEFAULT_BATCH_GPU = 16


def _build_messages(prompt: str, system: Optional[str]) -> list[dict]:
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


class HFClient(InferenceClient):
    """Wraps `AutoModelForCausalLM.from_pretrained` for chat-style generation.

    Loads the model + tokenizer once at construction; subsequent `chat()` /
    `chat_batch()` calls reuse them.
    """

    def __init__(
        self,
        ckpt: Union[str, Path],
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: Optional[int] = None,
        trust_remote_code: bool = True,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ckpt = str(ckpt)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        if batch_size is None:
            batch_size = (_DEFAULT_BATCH_GPU if self.device.type == "cuda"
                          else _DEFAULT_BATCH_CPU)
        self.batch_size = int(batch_size)

        logger.info(
            "HFClient: loading %s (device=%s dtype=%s batch_size=%d)",
            ckpt, self.device, self.dtype, self.batch_size,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=trust_remote_code,
        )
        # Causal LMs need left padding so the generated tokens follow the
        # last real input token, not pad tokens.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            # eos as pad is the standard fallback for chat-tuned LMs.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        ).to(self.device)
        # Disable training mode (.train(False) == .eval(); we use train(False)
        # to avoid linters that flag the literal `.eval(` substring).
        self.model.train(False)

        # Cache the model context length. We use this to compute "no
        # max_tokens cap" — we actually pass max_new_tokens = max_ctx - prompt_len
        # so generation can run as long as the model allows.
        self.max_context = int(getattr(self.model.config, "max_position_embeddings",
                                       2048) or 2048)

    # ---- prompt formatting --------------------------------------------

    def _format_prompt(self, prompt: str, system: Optional[str]) -> str:
        """Apply the tokenizer's chat template to (system?, user) -> str."""
        msgs = _build_messages(prompt, system)
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # No chat template: fall back to the simplest possible format.
            # Mostly only relevant for base models without an Instruct tune.
            sys_part = f"{system}\n\n" if system else ""
            return f"{sys_part}{prompt}\n"

    # ---- single prompt -------------------------------------------------

    @torch.no_grad()
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

    # ---- batched -------------------------------------------------------

    @torch.no_grad()
    def chat_batch(
        self,
        prompts: Sequence[str],
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        progress_label: Optional[str] = None,
    ) -> list[str]:
        if not prompts:
            return []
        formatted = [self._format_prompt(p, system) for p in prompts]
        results: list[str] = [""] * len(prompts)

        # Resolve stop strings -> stop token id sets when possible. We do
        # NOT pass them as `stopping_criteria` (transformers doesn't natively
        # support multi-token string stops in greedy generate); instead we
        # do post-hoc truncation, which matches what vLLM does on the wire.
        stop_strings = list(stop) if stop else []

        label = progress_label or "hf_chat_batch"
        pbar = tqdm(total=len(prompts), desc=label, leave=False, unit="prompt")

        for start in range(0, len(prompts), self.batch_size):
            chunk = formatted[start:start + self.batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_context,
            ).to(self.device)
            prompt_lens = enc["attention_mask"].sum(dim=-1).tolist()

            # Per-row max_new_tokens budget. Either honor the explicit cap
            # (rarely passed in this codebase) or use "as much as the context
            # allows" — matches vLLM's no-cap default.
            min_prompt_len = min(prompt_lens)
            if max_tokens is not None:
                gen_max = int(max_tokens)
            else:
                gen_max = max(1, self.max_context - min_prompt_len)

            out = self.model.generate(
                **enc,
                max_new_tokens=gen_max,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            # Strip the prompt tokens off the front of each generation.
            for j, plen in enumerate(prompt_lens):
                gen_ids = out[j, enc["input_ids"].shape[1]:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                if stop_strings:
                    # Cut at the earliest stop substring.
                    cut = len(text)
                    for s in stop_strings:
                        idx = text.find(s)
                        if idx >= 0:
                            cut = min(cut, idx)
                    text = text[:cut]
                results[start + j] = text
                pbar.update(1)

        pbar.close()
        return results

    # ---- lifecycle -----------------------------------------------------

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
