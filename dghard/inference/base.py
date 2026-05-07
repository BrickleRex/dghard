"""Inference client interface — uniform across vLLM and HF backends.

The benchmark scorers in `dghard.eval.benchmarks.*` only see this interface,
which is why we can swap backends with a single CLI flag while keeping the
same prompts and same scoring.
"""
from __future__ import annotations

import abc
from typing import Optional, Sequence


class InferenceClient(abc.ABC):
    """Generate assistant responses for chat prompts.

    Both `chat()` and `chat_batch()` apply the model's chat template to a
    user prompt (and optional system message) before generation, then
    return the decoded assistant message (no special tokens, no chat
    template artifacts).

    Implementations MUST:
      - use greedy decoding (temperature = 0).
      - NOT cap `max_new_tokens` artificially — let it run up to the
        model's max context (matching the paper's "no max_tokens" policy).
    """

    @abc.abstractmethod
    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str: ...

    @abc.abstractmethod
    def chat_batch(
        self,
        prompts: Sequence[str],
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        progress_label: Optional[str] = None,
    ) -> list[str]: ...

    def close(self) -> None:
        """Release any held resources. Default: no-op."""
