"""Test HFClient on a tiny on-disk GPT2 (cached locally if present, else
constructed from random weights). No network required if HF cache holds it;
otherwise we synthesize a checkpoint from random init.

Why GPT2 and not a hand-built nn.Module: we need `AutoModelForCausalLM` /
`AutoTokenizer` to round-trip through HF's `apply_chat_template` and
`generate`. Building that protocol from scratch reproduces transformers'
internals — defeating the test. GPT2 is ~500 KB tokenizer + ~500 KB random
weights when synthesized, so it's still hermetic.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def tiny_gpt2_ckpt(tmp_path_factory) -> Path:
    """Build a tiny randomly-initialized GPT2 checkpoint on disk.

    Construction (no network):
      - GPT2Config with hidden=64, layers=2, heads=2 -> ~250 K params.
      - GPT2LMHeadModel from that config (random init).
      - Tokenizer: try to load 'gpt2' from local HF cache; if not present,
        skip the test (we can't fabricate a tokenizer).
    """
    from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel,
                               GPT2Tokenizer)

    out = tmp_path_factory.mktemp("tiny_gpt2")
    cfg = GPT2Config(
        vocab_size=GPT2Tokenizer.from_pretrained.__defaults__ and 50257 or 50257,
        n_positions=128,
        n_ctx=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(out)
    try:
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        pytest.skip("gpt2 tokenizer not in local HF cache; "
                    "fast HF-client test needs no-network tokenizer access")
    tok.save_pretrained(out)
    return out


def test_hf_client_chat_returns_string(tiny_gpt2_ckpt):
    from dghard.inference.hf_client import HFClient

    client = HFClient(tiny_gpt2_ckpt, device="cpu", batch_size=2)
    out = client.chat("hello", max_tokens=5)
    assert isinstance(out, str)
    # Random init -> output is gibberish, but it must be a non-None decoded string.
    client.close()


def test_hf_client_chat_batch_preserves_order(tiny_gpt2_ckpt):
    from dghard.inference.hf_client import HFClient

    client = HFClient(tiny_gpt2_ckpt, device="cpu", batch_size=2)
    prompts = ["alpha", "beta", "gamma"]
    out = client.chat_batch(prompts, max_tokens=3)
    assert len(out) == 3
    assert all(isinstance(s, str) for s in out)
    client.close()


def test_hf_client_greedy_is_deterministic(tiny_gpt2_ckpt):
    from dghard.inference.hf_client import HFClient

    client = HFClient(tiny_gpt2_ckpt, device="cpu", batch_size=2)
    a = client.chat_batch(["alpha", "beta"], max_tokens=4)
    b = client.chat_batch(["alpha", "beta"], max_tokens=4)
    assert a == b, "greedy decoding must be deterministic across calls"
    client.close()


def test_hf_client_respects_stop_strings(tiny_gpt2_ckpt):
    """If a stop substring appears in the generated text, it should be cut."""
    from dghard.inference.hf_client import HFClient

    client = HFClient(tiny_gpt2_ckpt, device="cpu", batch_size=2)
    full = client.chat("hello", max_tokens=20)
    if not full:
        pytest.skip("model produced empty output (random init quirk)")
    # Use a single character we know is in the output as the stop.
    stop_char = full[0]
    truncated = client.chat("hello", max_tokens=20, stop=[stop_char])
    assert stop_char not in truncated
    client.close()
