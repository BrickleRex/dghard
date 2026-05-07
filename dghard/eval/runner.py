"""Per-checkpoint eval orchestrator.

Loads each named benchmark, formats prompts, dispatches them through the
inference client, scores the predictions, writes per-benchmark JSON.

Skip-on-existing: if `out_dir/<benchmark>.json` already exists, that
benchmark is treated as cached unless `force=True`. Lets reviewers Ctrl-C
mid-run and pick up where they left off.

A failure on one benchmark records `{"error": ...}` and continues with the
rest — one bad dataset shouldn't kill the whole eval.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from dghard.eval.benchmarks import get_benchmark
from dghard.eval.benchmarks.base import BenchmarkResult
from dghard.inference.base import InferenceClient

logger = logging.getLogger(__name__)


def _run_one_benchmark(
    name: str,
    client: InferenceClient,
    out_dir: Path,
    *,
    n_samples: Optional[int],
    cache_dir: Optional[Path],
    force: bool,
) -> tuple[dict, Path, list[BenchmarkResult]]:
    out_path = out_dir / f"{name}.json"
    if out_path.exists() and not force:
        cached = json.loads(out_path.read_text())
        logger.info("[%s] cached at %s (use --force to rerun)", name, out_path)
        return cached["summary"], out_path, []

    benchmark = get_benchmark(name)
    logger.info("[%s] loading dataset (n_samples=%s) ...", name, n_samples)
    rows = benchmark.load(
        subset_size=n_samples,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    prompts = [benchmark.format_prompt(r) for r in rows]
    logger.info("[%s] %d prompts queued", name, len(prompts))

    t0 = time.monotonic()
    preds = client.chat_batch(
        prompts,
        system=benchmark.system_prompt(),
        progress_label=name,
    )
    elapsed = time.monotonic() - t0

    sample_results: list[BenchmarkResult] = [
        benchmark.score(p, r) for p, r in zip(preds, rows)
    ]
    summary = benchmark.aggregate(sample_results)
    summary.update({
        "n_prompts": len(prompts),
        "elapsed_s": elapsed,
        "benchmark": name,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": summary,
        "samples": [
            {"pred_raw": r.pred_raw, "pred": r.pred_normalized,
             "gold": r.gold, "correct": r.correct, "meta": r.meta}
            for r in sample_results
        ],
    }, indent=2, ensure_ascii=False))
    logger.info("[%s] score=%.4f n=%d elapsed=%.1fs -> %s",
                name, summary.get("score", 0.0), summary.get("n", 0),
                elapsed, out_path)
    return summary, out_path, sample_results


def run_evaluation(
    ckpt_path: Path,
    benchmarks: Sequence[str],
    client: InferenceClient,
    out_dir: Path,
    *,
    n_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> dict[str, dict]:
    """Run each named benchmark against `client`. Returns {name: summary}.

    `ckpt_path` is recorded in the per-benchmark JSON metadata. The client
    is assumed to already be loaded with the correct checkpoint.

    Failed benchmarks are recorded as `{"error": "<message>"}` instead of
    a normal summary; the rest of the suite continues.
    """
    out_dir = Path(out_dir)
    out: dict[str, dict] = {}
    for name in benchmarks:
        try:
            summary, _path, _samples = _run_one_benchmark(
                name, client, out_dir,
                n_samples=n_samples, cache_dir=cache_dir, force=force,
            )
            summary["ckpt_path"] = str(ckpt_path)
            out[name] = summary
        except Exception as e:  # noqa: BLE001 — record + continue
            logger.exception("[%s] FAILED: %s", name, e)
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out
