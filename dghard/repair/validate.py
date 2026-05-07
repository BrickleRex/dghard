"""Pre-flight checks before any repair compute.

Catches the common reviewer mistakes — wrong base/FT pair, mismatched
architectures, shape drift — fast and with a clear error.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


class ValidationError(RuntimeError):
    """Raised when the (base, ft) pair cannot be DG-Hard-repaired."""


@dataclass
class ValidationReport:
    base: str
    ft: str
    model_type: Optional[str] = None
    architectures: Optional[str] = None
    n_matched_2d: int = 0
    n_passthrough_1d: int = 0
    n_unique_to_base: int = 0
    n_unique_to_ft: int = 0
    n_shape_mismatched: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_config_dict(path: Path) -> dict:
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        raise ValidationError(f"missing config.json at {cfg_path}")
    return json.loads(cfg_path.read_text())


def _safetensors_keys_and_shapes(ckpt_dir: Path) -> dict[str, tuple[int, ...]]:
    """Read every key + shape from every *.safetensors shard in `ckpt_dir`.

    Streams the shard headers — does NOT load tensor data.
    """
    from safetensors import safe_open

    shards = sorted(ckpt_dir.glob("*.safetensors"))
    if not shards:
        raise ValidationError(
            f"no *.safetensors files under {ckpt_dir}. "
            f"Only safetensors checkpoints are supported."
        )
    out: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                out[k] = tuple(f.get_slice(k).get_shape())
    return out


def validate_checkpoints(base_dir: Path, ft_dir: Path) -> ValidationReport:
    """Validate that DG-Hard can run on the (base, ft) pair.

    Hard failures (raise ValidationError):
      - Either path missing.
      - Either path missing config.json.
      - Either path has no *.safetensors shards.
      - model_type or architectures differ.

    Soft warnings (recorded in report.warnings, do not raise):
      - hidden_size / num_hidden_layers / vocab_size differ.
      - Keys present in only one side.
      - Shapes differ for shared keys (those keys are skipped at repair time).
    """
    base_dir = Path(base_dir)
    ft_dir = Path(ft_dir)
    if not base_dir.is_dir():
        raise ValidationError(f"base path is not a directory: {base_dir}")
    if not ft_dir.is_dir():
        raise ValidationError(f"ft path is not a directory: {ft_dir}")

    base_cfg = _load_config_dict(base_dir)
    ft_cfg = _load_config_dict(ft_dir)

    model_type = base_cfg.get("model_type")
    if base_cfg.get("model_type") != ft_cfg.get("model_type"):
        raise ValidationError(
            f"model_type mismatch: base={base_cfg.get('model_type')!r} "
            f"ft={ft_cfg.get('model_type')!r}"
        )
    if base_cfg.get("architectures") != ft_cfg.get("architectures"):
        raise ValidationError(
            f"architectures mismatch: base={base_cfg.get('architectures')!r} "
            f"ft={ft_cfg.get('architectures')!r}"
        )
    arch = ",".join(base_cfg.get("architectures") or [])

    warnings: list[str] = []
    for key in ("hidden_size", "num_hidden_layers", "vocab_size"):
        if base_cfg.get(key) != ft_cfg.get(key):
            warnings.append(
                f"config.{key} differs: base={base_cfg.get(key)} "
                f"ft={ft_cfg.get(key)}"
            )

    base_keys = _safetensors_keys_and_shapes(base_dir)
    ft_keys = _safetensors_keys_and_shapes(ft_dir)

    only_base = set(base_keys) - set(ft_keys)
    only_ft = set(ft_keys) - set(base_keys)
    shared = set(base_keys) & set(ft_keys)

    n_matched_2d = 0
    n_passthrough_1d = 0
    n_shape_mismatched = 0
    for k in shared:
        if base_keys[k] != ft_keys[k]:
            n_shape_mismatched += 1
            warnings.append(
                f"shape mismatch on {k}: base={base_keys[k]} ft={ft_keys[k]} "
                f"(will be skipped during repair)"
            )
            continue
        if len(base_keys[k]) == 2:
            n_matched_2d += 1
        else:
            n_passthrough_1d += 1

    if only_base:
        warnings.append(
            f"{len(only_base)} key(s) only in base "
            f"(will be passed through unchanged)"
        )
    if only_ft:
        warnings.append(
            f"{len(only_ft)} key(s) only in ft (ignored)"
        )
    if n_matched_2d == 0:
        raise ValidationError(
            "no shared 2D weight tensors between base and ft — "
            "nothing for DG-Hard to repair."
        )

    return ValidationReport(
        base=str(base_dir),
        ft=str(ft_dir),
        model_type=model_type,
        architectures=arch,
        n_matched_2d=n_matched_2d,
        n_passthrough_1d=n_passthrough_1d,
        n_unique_to_base=len(only_base),
        n_unique_to_ft=len(only_ft),
        n_shape_mismatched=n_shape_mismatched,
        warnings=warnings,
    )
