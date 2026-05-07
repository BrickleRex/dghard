"""ARC Challenge — AI2 Reasoning Challenge, hard subset (0-shot, MC)."""
from __future__ import annotations

import random
from typing import Optional

from dghard.eval.benchmarks.base import Benchmark, BenchmarkResult
from dghard.eval.utils import extract_letter


def _normalize_answer_key(raw: str, label_pool: list[str]) -> str:
    raw = str(raw).strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(label_pool):
            return label_pool[idx]
        return raw
    return raw.upper()


def _format_one_example(row: dict, with_answer: bool) -> str:
    q = row["question"].strip()
    choices = row["choices"]
    labels = list(choices["label"])
    texts = list(choices["text"])
    lines = [f"Question: {q}"]
    for lbl, txt in zip(labels, texts):
        lines.append(f"{lbl}. {txt}")
    if with_answer:
        gold = _normalize_answer_key(row["answerKey"], labels)
        lines.append(f"Answer: {gold}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def build_arc_prompt(eval_row: dict, fewshot_rows: list[dict]) -> str:
    parts = [
        "The following are multiple choice science questions. Answer with "
        "the letter of the correct option. Think step by step if needed, "
        "then output the answer on the final line."
    ]
    for r in fewshot_rows:
        parts.append(_format_one_example(r, with_answer=True))
    parts.append(_format_one_example(eval_row, with_answer=False))
    return "\n\n".join(parts)


class ArcChallenge(Benchmark):
    name = "arc_challenge"
    metric = "accuracy"

    def __init__(self, n_shot: int = 0, hf_config: str = "ARC-Challenge") -> None:
        self.n_shot = int(n_shot)
        self.hf_config = hf_config
        self._fewshot_pool: list[dict] = []

    def load(self, subset_size: Optional[int] = None, seed: int = 42,
             cache_dir=None):
        from datasets import load_dataset
        test_ds = load_dataset("allenai/ai2_arc", self.hf_config,
                               split="test", cache_dir=cache_dir)
        rows = [dict(r) for r in test_ds]
        if self.n_shot > 0:
            train_ds = load_dataset("allenai/ai2_arc", self.hf_config,
                                    split="train", cache_dir=cache_dir)
            self._fewshot_pool = [dict(r) for r in train_ds][: self.n_shot]
        else:
            self._fewshot_pool = []
        if subset_size and subset_size < len(rows):
            rng = random.Random(seed)
            rng.shuffle(rows)
            rows = rows[:subset_size]
        return rows

    def format_prompt(self, row: dict) -> str:
        return build_arc_prompt(row, self._fewshot_pool)

    def score(self, pred: str, row: dict) -> BenchmarkResult:
        labels = list(row["choices"]["label"])
        gold_letter = _normalize_answer_key(row["answerKey"], labels)
        pred_letter = (extract_letter(pred, choices="ABCDE") or "").upper()
        return BenchmarkResult(
            correct=float(pred_letter == gold_letter),
            pred_raw=pred,
            pred_normalized=pred_letter,
            gold=gold_letter,
            meta={"id": row.get("id")},
        )
