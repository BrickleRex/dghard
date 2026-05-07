"""MNLI — Multi-Genre Natural Language Inference (3-way classification)."""
from __future__ import annotations

import random
import re
from typing import Optional

from dghard.eval.benchmarks.base import Benchmark, BenchmarkResult
from dghard.eval.utils import strip_think_tags


_LABELS = ("entailment", "neutral", "contradiction")
_LETTER_TO_LABEL = {"A": "entailment", "B": "neutral", "C": "contradiction"}


def build_mnli_prompt(premise: str, hypothesis: str) -> str:
    return (
        "Determine the relationship between the premise and the hypothesis. "
        "Answer with one letter:\n"
        "A. entailment (the premise implies the hypothesis)\n"
        "B. neutral (neither implies the other)\n"
        "C. contradiction (the premise contradicts the hypothesis)\n\n"
        f"Premise: {premise.strip()}\n"
        f"Hypothesis: {hypothesis.strip()}\n"
        "Answer:"
    )


def _extract_mnli_label(text: str) -> Optional[str]:
    if not text:
        return None
    t = strip_think_tags(text)
    lines = [ln.strip() for ln in t.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        m = re.search(r"\b([ABC])\b", last)
        if m:
            return _LETTER_TO_LABEL[m.group(1)]
    low = t.lower()
    for lbl in _LABELS:
        if re.search(rf"\b{lbl}\b", low):
            return lbl
    m = re.search(r"\b([ABC])\b", t)
    if m:
        return _LETTER_TO_LABEL[m.group(1)]
    return None


class Mnli(Benchmark):
    name = "mnli"
    metric = "accuracy"

    def __init__(self, split: str = "validation_matched") -> None:
        self.split = split

    def load(self, subset_size: Optional[int] = None, seed: int = 42,
             cache_dir=None):
        from datasets import load_dataset
        ds = load_dataset("nyu-mll/multi_nli", split=self.split,
                          cache_dir=cache_dir)
        rows = [dict(r) for r in ds if r.get("label") in (0, 1, 2)]
        if subset_size and subset_size < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, subset_size)
        return rows

    def format_prompt(self, row: dict) -> str:
        return build_mnli_prompt(row["premise"], row["hypothesis"])

    def score(self, pred: str, row: dict) -> BenchmarkResult:
        gold_label = _LABELS[int(row["label"])]
        pred_label = _extract_mnli_label(pred) or ""
        return BenchmarkResult(
            correct=float(pred_label == gold_label),
            pred_raw=pred,
            pred_normalized=pred_label,
            gold=gold_label,
            meta={"genre": row.get("genre")},
        )
