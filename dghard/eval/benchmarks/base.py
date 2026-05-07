"""Benchmark interface — every per-benchmark module subclasses this."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class BenchmarkResult:
    correct: float                 # in [0, 1]
    pred_raw: str
    pred_normalized: str
    gold: Any
    meta: dict[str, Any] = field(default_factory=dict)


class Benchmark(abc.ABC):
    name: str
    metric: str                    # e.g. 'accuracy', 'exact_number', 'f1_em'

    @abc.abstractmethod
    def load(self, subset_size: Optional[int] = None,
             seed: int = 42, cache_dir: Optional[str] = None) -> list[dict]:
        """Return a list of eval rows (full split if subset_size is None)."""

    @abc.abstractmethod
    def format_prompt(self, row: dict) -> str:
        """Build the user-message content for a row.

        The chat template is applied later by the inference client; this
        returns plain text only.
        """

    def system_prompt(self) -> Optional[str]:
        """Optional system message. Most benchmarks return None."""
        return None

    @abc.abstractmethod
    def score(self, pred: str, row: dict) -> BenchmarkResult:
        """Score a single (pred, row) pair."""

    def aggregate(self, results: Iterable[BenchmarkResult]) -> dict:
        results = list(results)
        if not results:
            return {"score": 0.0, "n": 0, "metric": self.metric}
        score = sum(r.correct for r in results) / len(results)
        return {"score": score, "n": len(results), "metric": self.metric}
