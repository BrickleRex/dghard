"""Benchmark registry. Each benchmark is lazy-imported to keep startup fast."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from dghard.eval.benchmarks.base import Benchmark


def _lazy(module: str, attr: str) -> Callable[[], "Benchmark"]:
    def factory() -> "Benchmark":
        import importlib
        mod = importlib.import_module(f"dghard.eval.benchmarks.{module}")
        return getattr(mod, attr)()
    return factory


BENCHMARKS: dict[str, Callable[[], "Benchmark"]] = {
    "gsm8k":         _lazy("gsm8k",         "Gsm8k"),
    "mmlu":          _lazy("mmlu",          "Mmlu"),
    "arc_challenge": _lazy("arc_challenge", "ArcChallenge"),
    "hellaswag":     _lazy("hellaswag",     "HellaSwag"),
    "truthful_qa":   _lazy("truthful_qa",   "TruthfulQa"),
    "ifeval":        _lazy("ifeval",        "IfEval"),
    "math_500":      _lazy("math_500",      "Math500"),
    "mnli":          _lazy("mnli",          "Mnli"),
    "trivia_qa":     _lazy("trivia_qa",     "TriviaQa"),
}


def get_benchmark(name: str) -> "Benchmark":
    if name not in BENCHMARKS:
        raise KeyError(
            f"Unknown benchmark: {name!r}. Available: {sorted(BENCHMARKS)}"
        )
    return BENCHMARKS[name]()
