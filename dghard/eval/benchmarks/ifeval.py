"""IFEval — instruction-following with rule-verifiable constraints.

We delegate scoring to the canonical Google verifier as packaged in
EleutherAI's `lm-evaluation-harness` (`lm_eval.tasks.ifeval.utils`). That
package vendors and maintains the official 25-rule registry, the strict /
loose evaluators, and the keyword-list data files. We aggregate to the four
standard metrics and return their average as the headline `score`.

`lm_eval` is heavy enough that we keep it OUT of the base install — install
it explicitly with `pip install lm_eval` to enable IFEval. If unavailable,
this benchmark raises a clear error at scoring time.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

from dghard.eval.benchmarks.base import Benchmark, BenchmarkResult

logger = logging.getLogger(__name__)


def _ensure_nltk_data() -> None:
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
    except Exception:
        logger.warning("[ifeval] could not pre-warm nltk punkt_tab")


class IfEval(Benchmark):
    name = "ifeval"
    metric = "accuracy"

    def load(self, subset_size: Optional[int] = None, seed: int = 42,
             cache_dir=None):
        from datasets import load_dataset
        _ensure_nltk_data()
        ds = load_dataset("google/IFEval", split="train", cache_dir=cache_dir)
        rows = [dict(r) for r in ds]
        if subset_size and subset_size < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, subset_size)
        return rows

    def format_prompt(self, row: dict) -> str:
        return row["prompt"]

    def score(self, pred: str, row: dict) -> BenchmarkResult:
        try:
            from lm_eval.tasks.ifeval.utils import (
                InputExample,
                test_instruction_following_loose,
                test_instruction_following_strict,
            )
        except ImportError as e:
            raise RuntimeError(
                "IFEval needs the EleutherAI lm-evaluation-harness for the "
                "official Google rule verifier. Install with "
                "`pip install lm_eval` then retry."
            ) from e

        inp = InputExample(
            key=row.get("key"),
            instruction_id_list=list(row["instruction_id_list"]),
            prompt=row["prompt"],
            kwargs=list(row["kwargs"] or []),
        )
        out_strict = test_instruction_following_strict(inp, pred)
        out_loose = test_instruction_following_loose(inp, pred)
        n = max(1, len(inp.instruction_id_list))
        prompt_strict = float(out_strict.follow_all_instructions)
        prompt_loose = float(out_loose.follow_all_instructions)
        inst_strict = sum(out_strict.follow_instruction_list) / n
        inst_loose = sum(out_loose.follow_instruction_list) / n
        score = (prompt_strict + prompt_loose + inst_strict + inst_loose) / 4.0
        return BenchmarkResult(
            correct=score,
            pred_raw=pred,
            pred_normalized="",
            gold="",
            meta={
                "instruction_ids": list(inp.instruction_id_list),
                "prompt_level_strict_acc": prompt_strict,
                "prompt_level_loose_acc": prompt_loose,
                "inst_level_strict_acc": inst_strict,
                "inst_level_loose_acc": inst_loose,
            },
        )
