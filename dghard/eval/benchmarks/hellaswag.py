"""HellaSwag — commonsense sentence-completion (4-choice MC, 0-shot)."""
from __future__ import annotations

import random
from typing import Optional

from dghard.eval.benchmarks.base import Benchmark, BenchmarkResult
from dghard.eval.utils import extract_letter


def _letter(idx: int) -> str:
    return "ABCD"[idx]


def _ctx(row: dict) -> str:
    if row.get("ctx"):
        return row["ctx"].strip()
    a = (row.get("ctx_a") or "").strip()
    b = (row.get("ctx_b") or "").strip()
    return f"{a} {b}".strip() if b else a


def build_hellaswag_prompt(row: dict) -> str:
    activity = (row.get("activity_label") or "").strip()
    ctx = _ctx(row)
    endings = list(row["endings"])
    lines = [
        "Choose the most plausible continuation of the passage. Answer with "
        "the letter A, B, C, or D.",
    ]
    if activity:
        lines.append(f"Activity: {activity}")
    lines.append(f"Passage: {ctx}")
    for i, e in enumerate(endings):
        lines.append(f"{_letter(i)}. {e.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


class HellaSwag(Benchmark):
    name = "hellaswag"
    metric = "accuracy"

    def load(self, subset_size: Optional[int] = None, seed: int = 42,
             cache_dir=None):
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation",
                          cache_dir=cache_dir)
        rows = [dict(r) for r in ds]
        if subset_size and subset_size < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, subset_size)
        return rows

    def format_prompt(self, row: dict) -> str:
        return build_hellaswag_prompt(row)

    def score(self, pred: str, row: dict) -> BenchmarkResult:
        gold_idx = int(row["label"])
        gold_letter = _letter(gold_idx)
        pred_letter = (extract_letter(pred) or "").upper()
        return BenchmarkResult(
            correct=float(pred_letter == gold_letter),
            pred_raw=pred,
            pred_normalized=pred_letter,
            gold=gold_letter,
            meta={"activity_label": row.get("activity_label")},
        )
