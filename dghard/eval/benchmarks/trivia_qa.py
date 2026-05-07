"""TriviaQA (rc.nocontext) — closed-book trivia QA.

Closed-book — no retrieved evidence — so a clean parametric-knowledge
forgetting signal. SQuAD-style normalization, hit if pred matches any gold
alias.
"""
from __future__ import annotations

import random
import re
import string
from typing import Optional

from dghard.eval.benchmarks.base import Benchmark, BenchmarkResult
from dghard.eval.utils import strip_think_tags


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_TBL = str.maketrans("", "", string.punctuation)


def _normalize(s: str) -> str:
    s = s.lower()
    s = _ARTICLES_RE.sub(" ", s)
    s = s.translate(_PUNCT_TBL)
    s = " ".join(s.split())
    return s


def _extract_short_answer(text: str) -> str:
    if not text:
        return ""
    t = strip_think_tags(text).strip()
    low = t.lower()
    for marker in ("answer:", "the answer is", "answer is", "final answer:"):
        if marker in low:
            idx = low.rindex(marker) + len(marker)
            tail = t[idx:].strip().splitlines()
            if tail:
                return tail[0].strip().rstrip(".,;:!?")
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        return lines[-1].rstrip(".,;:!?")
    return t


def build_trivia_prompt(question: str) -> str:
    return (
        "Answer the trivia question with a short factual response. "
        "Output the answer on the final line after 'Answer:'.\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def _aliases(row: dict) -> list[str]:
    ans = row.get("answer") or {}
    out: list[str] = []
    for k in ("value", "normalized_value"):
        v = ans.get(k)
        if v:
            out.append(str(v))
    for k in ("aliases", "normalized_aliases"):
        seq = ans.get(k) or []
        out.extend(str(a) for a in seq)
    return out


class TriviaQa(Benchmark):
    name = "trivia_qa"
    metric = "accuracy"

    def __init__(self, hf_config: str = "rc.nocontext") -> None:
        self.hf_config = hf_config

    def load(self, subset_size: Optional[int] = None, seed: int = 42,
             cache_dir=None):
        from datasets import load_dataset
        ds = load_dataset("mandarjoshi/trivia_qa", self.hf_config,
                          split="validation", cache_dir=cache_dir)
        rows = [dict(r) for r in ds]
        if subset_size and subset_size < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, subset_size)
        return rows

    def format_prompt(self, row: dict) -> str:
        return build_trivia_prompt(row["question"])

    def score(self, pred: str, row: dict) -> BenchmarkResult:
        pred_short = _extract_short_answer(pred)
        pred_norm = _normalize(pred_short)
        golds = _aliases(row)
        gold_norms = {_normalize(g) for g in golds}
        gold_norms.discard("")
        correct = float(pred_norm in gold_norms) if pred_norm else 0.0
        return BenchmarkResult(
            correct=correct,
            pred_raw=pred,
            pred_normalized=pred_norm,
            gold=sorted(gold_norms),
            meta={"question_id": row.get("question_id")},
        )
