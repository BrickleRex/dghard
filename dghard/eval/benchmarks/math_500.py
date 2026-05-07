"""MATH-500 — 500-question subset of Hendrycks et al. (2021) MATH (0-shot)."""
from __future__ import annotations

import random
import re
from typing import Optional

from dghard.eval.benchmarks.base import Benchmark, BenchmarkResult
from dghard.eval.utils import strip_think_tags


def _strip_outer(s: str) -> str:
    s = s.strip()
    for _ in range(4):
        m = re.match(r"^\\(?:text|mathrm|displaystyle)\{(.*)\}$", s)
        if m:
            s = m.group(1).strip()
            continue
        if s.startswith("$") and s.endswith("$"):
            s = s[1:-1].strip()
            continue
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1].strip()
            continue
        break
    return s


def _canon(s: str) -> str:
    s = _strip_outer(s)
    s = s.replace(" ", "")
    s = s.replace(",", "")
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\\\", "\\")
    s = s.rstrip(".")
    return s


def _extract_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    text = strip_think_tags(text)
    last_start = text.rfind(r"\boxed{")
    if last_start < 0:
        last_start = text.rfind("boxed{")
        if last_start < 0:
            return None
        i = last_start + len("boxed{")
    else:
        i = last_start + len(r"\boxed{")
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
        i += 1
    if depth != 0:
        return None
    return "".join(out)


def extract_math_answer(text: str) -> Optional[str]:
    boxed = _extract_boxed(text)
    if boxed is not None:
        return _canon(boxed)
    if not text:
        return None
    low = text.lower()
    for marker in ("final answer:", "the answer is", "answer is", "answer:"):
        if marker in low:
            idx = low.rindex(marker) + len(marker)
            tail = text[idx:idx + 200].strip()
            tail = tail.split("\n", 1)[0].strip()
            return _canon(tail)
    return None


def build_math_prompt(question: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning, then put "
        "your final answer inside \\boxed{...}.\n\n"
        f"Problem: {question.strip()}"
    )


class Math500(Benchmark):
    name = "math_500"
    metric = "accuracy"

    def load(self, subset_size: Optional[int] = None, seed: int = 42,
             cache_dir=None):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test",
                          cache_dir=cache_dir)
        rows = [dict(r) for r in ds]
        if subset_size and subset_size < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, subset_size)
        return rows

    def format_prompt(self, row: dict) -> str:
        return build_math_prompt(row["problem"])

    def score(self, pred: str, row: dict) -> BenchmarkResult:
        gold_raw = str(row.get("answer", ""))
        gold = _canon(gold_raw)
        pred_norm = extract_math_answer(pred) or ""
        return BenchmarkResult(
            correct=float(pred_norm == gold and gold != ""),
            pred_raw=pred,
            pred_normalized=pred_norm,
            gold=gold,
            meta={"subject": row.get("subject"), "level": row.get("level")},
        )
