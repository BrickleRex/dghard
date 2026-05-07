"""MMLU — Massive Multitask Language Understanding (57 subjects, 4-choice).

5-shot prompting with examples drawn from each subject's `dev` split, per
the original Hendrycks et al. (2021) protocol. Scoring: letter extraction
with robust fallbacks. Ported verbatim from the paper-tree implementation.
"""
from __future__ import annotations

import random
from typing import Optional

from dghard.eval.benchmarks.base import Benchmark, BenchmarkResult
from dghard.eval.utils import extract_letter


def _letter(idx: int) -> str:
    return "ABCD"[idx]


def _format_one_example(row: dict, with_answer: bool) -> str:
    q = row["question"].strip()
    choices = row["choices"]
    lines = [f"Question: {q}"]
    for i, c in enumerate(choices):
        lines.append(f"{_letter(i)}. {c}")
    lines.append(f"Answer: {_letter(row['answer'])}" if with_answer else "Answer:")
    return "\n".join(lines)


def build_mmlu_prompt(eval_row: dict, fewshot_rows: list[dict]) -> str:
    parts = []
    subject = eval_row.get("subject", "a subject").replace("_", " ")
    parts.append(
        f"The following are multiple choice questions about {subject}. "
        "Answer with the letter A, B, C, or D. Think step by step if needed, "
        "then output the answer on the final line."
    )
    for r in fewshot_rows:
        parts.append(_format_one_example(r, with_answer=True))
    parts.append(_format_one_example(eval_row, with_answer=False))
    return "\n\n".join(parts)


class Mmlu(Benchmark):
    name = "mmlu"
    metric = "accuracy"

    def __init__(self, n_shot: int = 5, hf_config: str = "all") -> None:
        self.n_shot = int(n_shot)
        self.hf_config = hf_config
        self._dev_by_subject: dict[str, list[dict]] = {}

    def load(self, subset_size: Optional[int] = None, seed: int = 42,
             cache_dir=None):
        from datasets import load_dataset

        test_ds = load_dataset("cais/mmlu", self.hf_config, split="test",
                               cache_dir=cache_dir)
        dev_ds = load_dataset("cais/mmlu", self.hf_config, split="dev",
                              cache_dir=cache_dir)
        self._dev_by_subject.clear()
        for r in dev_ds:
            self._dev_by_subject.setdefault(r["subject"], []).append(dict(r))

        rows = [dict(r) for r in test_ds]
        rng = random.Random(seed)
        if subset_size and subset_size < len(rows):
            # Stratified subsetting: floor allocation per subject + one extra
            # for the first (N % S) subjects so every subject is represented.
            by_subject: dict[str, list[dict]] = {}
            for r in rows:
                by_subject.setdefault(r["subject"], []).append(r)
            subjects = sorted(by_subject)
            n_subj = len(subjects)
            base, extra = divmod(subset_size, n_subj)
            sampled = []
            for i, s in enumerate(subjects):
                bucket = by_subject[s]
                rng.shuffle(bucket)
                quota = base + (1 if i < extra else 0)
                sampled.extend(bucket[:quota])
            rows = sampled
        return rows

    def format_prompt(self, row: dict) -> str:
        subject = row["subject"]
        dev_pool = self._dev_by_subject.get(subject, [])
        fewshot = dev_pool[: self.n_shot]
        return build_mmlu_prompt(row, fewshot)

    def score(self, pred: str, row: dict) -> BenchmarkResult:
        gold_letter = _letter(row["answer"])
        pred_letter = extract_letter(pred) or ""
        return BenchmarkResult(
            correct=float(pred_letter == gold_letter),
            pred_raw=pred,
            pred_normalized=pred_letter,
            gold=gold_letter,
            meta={"subject": row.get("subject")},
        )
