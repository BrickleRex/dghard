"""End-to-end round-trip on a real (tiny) HuggingFace model.

Downloads ~270 MB on first run; cached afterwards. Marked `slow` so the
default `pytest` command skips it. Run explicitly with:

    pytest -m slow tests/integration/

What it covers:
  1. Load `HuggingFaceTB/SmolLM2-135M-Instruct` from HF.
  2. Snapshot it as a fixed "base" checkpoint dir.
  3. Apply a tiny synthetic delta to every 2D weight to get a fake "ft" dir.
  4. Run `dghard repair` to produce `repaired/`.
  5. Reload `repaired/` via `AutoModelForCausalLM` — proves it's a valid HF
     checkpoint.
  6. Run `dghard eval --benchmarks gsm8k --n-samples 4 --inference hf` and
     assert the output JSON has a real `summary.score` in [0, 1].

Wall-clock target: < 5 min on CPU. The 4-sample GSM8K eval dominates — each
prompt is ~1k tokens of few-shot context, which the model has to digest
greedy on CPU.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow


SMOLLM = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def base_ckpt(tmp_path_factory):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = tmp_path_factory.mktemp("base")
    tok = AutoTokenizer.from_pretrained(SMOLLM)
    model = AutoModelForCausalLM.from_pretrained(SMOLLM, torch_dtype=torch.float32)
    tok.save_pretrained(out)
    model.save_pretrained(out, safe_serialization=True)
    return out


@pytest.fixture(scope="module")
def ft_ckpt(tmp_path_factory, base_ckpt):
    """Snapshot base, then add a small noise delta to every 2D weight."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = tmp_path_factory.mktemp("ft")
    # Copy non-weight files (config, tokenizer, etc.).
    for f in base_ckpt.iterdir():
        if f.suffix == ".safetensors":
            continue
        if f.is_file():
            shutil.copy2(f, out / f.name)

    g = torch.Generator().manual_seed(42)
    for shard in sorted(base_ckpt.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
        for k, t in sd.items():
            if t.dim() == 2:
                noise = 0.02 * torch.randn(*t.shape, generator=g,
                                           dtype=torch.float32).to(t.dtype)
                sd[k] = t + noise
        save_file({k: v.contiguous() for k, v in sd.items()},
                  str(out / shard.name))
    return out


def test_repair_produces_loadable_hf_checkpoint(tmp_path, base_ckpt, ft_ckpt):
    from transformers import AutoModelForCausalLM
    from dghard.cli import main as cli_main

    repaired = tmp_path / "repaired"
    rc = cli_main([
        "repair",
        "--base", str(base_ckpt),
        "--ft", str(ft_ckpt),
        "--out", str(repaired),
        "--device", "cpu",
    ])
    assert rc == 0
    assert (repaired / "model.safetensors").exists()
    assert (repaired / "config.json").exists()

    # Reload through HF — proves the output is a valid checkpoint.
    model = AutoModelForCausalLM.from_pretrained(repaired, torch_dtype=torch.float32)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_eval_runs_on_repaired_ckpt(tmp_path, base_ckpt, ft_ckpt):
    """Repair, then eval 4 GSM8K samples through the HF backend."""
    from dghard.cli import main as cli_main

    repaired = tmp_path / "repaired"
    cli_main([
        "repair", "--base", str(base_ckpt), "--ft", str(ft_ckpt),
        "--out", str(repaired), "--device", "cpu",
    ])

    out_dir = tmp_path / "eval_out"
    rc = cli_main([
        "eval",
        "--ckpt", str(repaired),
        "--benchmarks", "gsm8k",
        "--n-samples", "4",
        "--inference", "hf",
        "--out-dir", str(out_dir),
        "--batch-size", "2",
    ])
    assert rc == 0

    payload = json.loads((out_dir / "gsm8k.json").read_text())
    summary = payload["summary"]
    assert "score" in summary
    assert isinstance(summary["score"], float)
    assert 0.0 <= summary["score"] <= 1.0
    assert summary["n"] == 4
    assert summary["benchmark"] == "gsm8k"
    assert len(payload["samples"]) == 4
    for s in payload["samples"]:
        assert "pred_raw" in s and "gold" in s and "correct" in s
