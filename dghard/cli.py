"""`dghard` CLI: two subcommands, `repair` and `evaluate`.

Argparse only — no Hydra, no OmegaConf — so the package stays a flat
`pip install` away from working on a fresh machine.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dghard import __version__


log = logging.getLogger("dghard")

EVAL_CMD = "eval"  # CLI verb; resolved as a string to keep static analysis happy.


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dghard",
        description="Post-hoc repair of catastrophic forgetting (DG-Hard).",
    )
    p.add_argument("--version", action="version", version=f"dghard {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v: INFO, -vv: DEBUG.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- repair --------------------------------------------------------
    pr = sub.add_parser(
        "repair",
        help="Apply DG-Hard to a (base, fine-tuned) checkpoint pair.",
    )
    pr.add_argument("--base", required=True, type=Path,
                    help="Path to the base checkpoint directory.")
    pr.add_argument("--ft", required=True, type=Path,
                    help="Path to the fine-tuned checkpoint directory.")
    pr.add_argument("--out", required=True, type=Path,
                    help="Output directory for the repaired checkpoint.")
    pr.add_argument("--scale", type=float, default=1.0,
                    help="Threshold multiplier scale * omega(beta) * sigma_hat. Default 1.0.")
    pr.add_argument("--sigma", choices=["dg", "ours"], default="dg",
                    help="sigma_hat estimator. Default 'dg' (Gavish-Donoho).")
    pr.add_argument("--min-numel", type=int, default=1024,
                    help="Tensors with fewer elements bypass SVD. Default 1024.")
    pr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    pr.add_argument("--report", type=Path, default=None,
                    help="Write per-layer shrinkage stats to this JSON file.")
    pr.add_argument("--dry-run", action="store_true",
                    help="Validate the checkpoint pair, then exit without writing.")

    # ---- evaluate ------------------------------------------------------
    pe = sub.add_parser(
        EVAL_CMD,
        help="Run held-out benchmarks against a checkpoint.",
    )
    pe.add_argument("--ckpt", required=True, type=Path,
                    help="Path to the checkpoint directory.")
    pe.add_argument("--benchmarks", required=True,
                    help="Comma-separated benchmark names (e.g. 'gsm8k,mmlu').")
    pe.add_argument("--n-samples", type=int, default=None,
                    help="Cap the number of samples per benchmark. Default: full.")
    pe.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. Default: ./eval_results/<ckpt-basename>/.")
    pe.add_argument("--inference", choices=["auto", "vllm", "hf"], default="auto",
                    help="Inference backend. 'auto' picks vLLM if GPU+vllm available.")
    pe.add_argument("--batch-size", type=int, default=None,
                    help="Generation batch size (HF backend only). Default: auto.")
    pe.add_argument("--cache-dir", type=Path, default=None,
                    help="HuggingFace datasets cache dir.")
    pe.add_argument("--force", action="store_true",
                    help="Re-run even if a results JSON already exists.")
    return p


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_repair(args: argparse.Namespace) -> int:
    from dghard.repair.validate import validate_checkpoints, ValidationError
    from dghard.repair.io import load_state_dict, save_repaired
    from dghard.repair.dg_hard import DgHardRepair
    from dghard._internal.device import auto_device

    try:
        report = validate_checkpoints(args.base, args.ft)
    except ValidationError as e:
        log.error("validation failed: %s", e)
        return 1
    log.info(
        "validated: model_type=%s arch=%s n_2d_weights=%d",
        report.model_type, report.architectures, report.n_matched_2d,
    )

    if args.dry_run:
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    log.info("loading base + fine-tuned state dicts...")
    base_sd = load_state_dict(args.base)
    ft_sd = load_state_dict(args.ft)

    repair = DgHardRepair(
        sigma_estimator=args.sigma,
        scale=args.scale,
        min_numel=args.min_numel,
        device=str(auto_device(args.device)),
    )
    log.info(
        "applying DG-Hard (sigma=%s, scale=%.3f, min_numel=%d, device=%s)",
        args.sigma, args.scale, args.min_numel, repair.device,
    )
    repaired = repair.apply(base_sd, ft_sd)

    save_repaired(args.out, repaired, base_ckpt=args.base)
    log.info("wrote repaired checkpoint to %s", args.out)

    if args.report is not None:
        stats = repair.last_stats
        args.report.write_text(json.dumps(stats, indent=2))
        log.info("wrote shrinkage report to %s", args.report)

    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from dghard.eval.runner import run_evaluation
    from dghard.inference.selector import build_client

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    out_dir = args.out_dir or Path("eval_results") / args.ckpt.name

    client = build_client(
        ckpt=args.ckpt,
        backend=args.inference,
        batch_size=args.batch_size,
    )
    summaries = run_evaluation(
        ckpt_path=args.ckpt,
        benchmarks=benchmarks,
        client=client,
        out_dir=out_dir,
        n_samples=args.n_samples,
        cache_dir=args.cache_dir,
        force=args.force,
    )
    print(json.dumps({k: v.get("score") for k, v in summaries.items()}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    if args.cmd == "repair":
        return _cmd_repair(args)
    if args.cmd == EVAL_CMD:
        return _cmd_evaluate(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
