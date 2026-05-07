"""Tests for the eval runner: skip/force, JSON schema, error isolation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import pytest

from dghard.eval.runner import run_evaluation
from dghard.inference.base import InferenceClient


class _FakeClient(InferenceClient):
    """Returns a fixed string for every prompt; counts how many calls happen."""

    def __init__(self, response: str = "Answer: A"):
        self.response = response
        self.call_count = 0

    def chat(self, prompt, *, system=None, max_tokens=None, stop=None):
        self.call_count += 1
        return self.response

    def chat_batch(self, prompts, *, system=None, max_tokens=None, stop=None,
                   progress_label=None):
        self.call_count += len(prompts)
        return [self.response] * len(prompts)


@pytest.fixture
def fake_dataset(monkeypatch):
    """Mock `datasets.load_dataset` for a couple of benchmarks so the
    runner can be tested without network.

    We monkeypatch the `datasets.load_dataset` global before benchmarks
    are constructed.
    """
    # ARC and HellaSwag both call `datasets.load_dataset` inside `load`.
    # We provide canned datasets that return small lists.

    class _FakeIterable(list):
        """Quacks enough like a HF Dataset for our consumption."""
        def __iter__(self):
            return list.__iter__(self)

    arc_rows = [
        {"question": "What gas do plants absorb?",
         "choices": {"label": ["A", "B", "C", "D"],
                     "text": ["O2", "CO2", "N2", "He"]},
         "answerKey": "A",     # Wrong (gold should be B); fake is just a fixture.
         "id": "1"},
        {"question": "Color of the sun?",
         "choices": {"label": ["A", "B", "C", "D"],
                     "text": ["red", "yellow", "blue", "green"]},
         "answerKey": "A", "id": "2"},
    ]

    def fake_load_dataset(repo, *args, **kwargs):
        if repo == "allenai/ai2_arc":
            return _FakeIterable(arc_rows)
        raise RuntimeError(f"unexpected dataset request: {repo}")

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return arc_rows


def test_runner_writes_summary_and_samples(tmp_path, fake_dataset):
    client = _FakeClient(response="Answer: A")
    out_dir = tmp_path / "results"
    summaries = run_evaluation(
        ckpt_path=tmp_path / "fake_ckpt",
        benchmarks=["arc_challenge"],
        client=client,
        out_dir=out_dir,
    )
    assert "arc_challenge" in summaries
    s = summaries["arc_challenge"]
    assert "score" in s and "n" in s and s["n"] == 2

    payload = json.loads((out_dir / "arc_challenge.json").read_text())
    assert "summary" in payload
    assert "samples" in payload
    assert len(payload["samples"]) == 2
    for sample in payload["samples"]:
        assert "pred_raw" in sample and "pred" in sample
        assert "gold" in sample and "correct" in sample


def test_runner_skips_existing_results(tmp_path, fake_dataset):
    client = _FakeClient()
    out_dir = tmp_path / "results"
    run_evaluation(tmp_path / "ckpt", ["arc_challenge"], client, out_dir)
    n_calls_first = client.call_count

    # Second run: no extra calls because the JSON exists.
    run_evaluation(tmp_path / "ckpt", ["arc_challenge"], client, out_dir)
    assert client.call_count == n_calls_first


def test_runner_force_reruns(tmp_path, fake_dataset):
    client = _FakeClient()
    out_dir = tmp_path / "results"
    run_evaluation(tmp_path / "ckpt", ["arc_challenge"], client, out_dir)
    first_calls = client.call_count
    run_evaluation(tmp_path / "ckpt", ["arc_challenge"], client, out_dir,
                   force=True)
    assert client.call_count == 2 * first_calls


def test_runner_isolates_failures_per_benchmark(tmp_path):
    """Unknown benchmark name -> recorded as error, doesn't crash the whole call."""
    client = _FakeClient()
    summaries = run_evaluation(
        ckpt_path=tmp_path / "ckpt",
        benchmarks=["bogus_benchmark_name"],
        client=client,
        out_dir=tmp_path / "results",
    )
    assert "bogus_benchmark_name" in summaries
    assert "error" in summaries["bogus_benchmark_name"]
