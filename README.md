# dghard — Spectral Unforgetting: Post-Hoc Recovery of Damaged Capabilities Without Retraining

Reference implementation of **DG-Hard**, the method introduced in *Spectral Unforgetting:
Post-Hoc Recovery of Damaged Capabilities Without Retraining* (NeurIPS 2026). DG-Hard takes a base
checkpoint θ₀ and a fine-tuned checkpoint θₙ of the same architecture, computes the
fine-tune delta Δ = θₙ − θ₀, and returns θ\* = θ₀ + Δ\* by hard-thresholding the
singular values of Δ at the Donoho-Gavish optimal cutoff τ\* = ω(β)·σ̂. No training,
no fine-tuning data, no hyperparameter search.

## Install

```bash
pip install -e .                    # CPU-only / HF generate
pip install -e ".[gpu]"             # also installs vLLM for GPU inference
pip install -e ".[dev]"             # adds pytest
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

## Repair a checkpoint

```bash
dghard repair --base path/to/base_ckpt \
              --ft   path/to/finetuned_ckpt \
              --out  path/to/repaired_ckpt
```

The output directory is a standard HuggingFace checkpoint that loads via
`AutoModelForCausalLM.from_pretrained(...)`. The base model's tokenizer and config
are copied alongside the repaired weights.

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--scale` | `1.0` | Multiplier on the threshold τ\* = scale · ω(β) · σ̂. The paper uses 1.0 (parameter-free Donoho-Gavish edge). |
| `--sigma` | `dg` | σ̂ estimator. `dg` = aspect-aware MP median; `ours` = Q25-MAD tail. |
| `--min-numel` | `1024` | Tensors smaller than this (or 1D — biases, norms) are passed through. |
| `--device` | `auto` | `cpu`, `cuda`, or `auto`. SVD is computed in fp32 then cast back. |
| `--report` | — | Path to write a per-layer shrinkage stats JSON (schema below). |
| `--dry-run` | off | Validate only — no SVD, no write. Prints the validation report as JSON. |

### Output layout

The repaired directory loads via `AutoModelForCausalLM.from_pretrained(path)`:

```
<out>/
├── model.safetensors        # repaired weights, single shard
├── config.json              # copied from --base
├── tokenizer.json           # copied from --base (and other tokenizer files)
└── dghard_repair.json       # marker: { method, base_ckpt, n_tensors }
```

### Shrinkage report schema (`--report`)

```jsonc
{
  "<param.name.weight>": {
    "shape":          [m, n],          // matrix dims
    "rank_total":     467,             // min(m, n)
    "rank_kept":      21,              // singular values above τ*
    "sigma_hat":      0.0042,          // estimated noise scale (fused σ_n·√n_large)
    "threshold":      0.0097,          // scale · ω(β) · sigma_hat
    "frob_delta_in":  3.21,            // ||Δ_FT||_F
    "frob_delta_out": 1.84             // ||Δ*||_F  (after shrinkage)
  },
  ...
}
```

One entry per 2D weight that the SVD touched. Tensors that bypassed SVD
(1D, or below `--min-numel`) don't appear here.

### Validation report schema (`--dry-run`)

```jsonc
{
  "base":               "/path/to/base",
  "ft":                 "/path/to/ft",
  "model_type":         "qwen3",
  "architectures":      "Qwen3ForCausalLM",
  "n_matched_2d":       197,    // 2D weights repair will touch
  "n_passthrough_1d":   113,    // 1D weights (biases, layer norms) — skipped
  "n_unique_to_base":   0,
  "n_unique_to_ft":     0,
  "n_shape_mismatched": 0,
  "warnings":           []
}
```

Exit code is `1` on any blocker (missing config, model_type / architectures
mismatch, no shared 2D weights). Warnings (extra keys, shape drift on
shared keys, hidden_size differs) are reported in `warnings` but don't
fail.

## Evaluate a checkpoint

```bash
dghard eval --ckpt path/to/ckpt \
            --benchmarks gsm8k,mmlu,arc_challenge \
            --n-samples 200
```

Available benchmarks: `gsm8k`, `mmlu`, `arc_challenge`, `hellaswag`, `truthful_qa`,
`ifeval`, `math_500`, `mnli`, `trivia_qa`. Datasets are downloaded on first use via
`datasets.load_dataset()` and cached under `$HF_HOME/datasets`.

Inference backend is auto-selected:
- GPU present + `vllm` installed → vLLM offline batched inference.
- Otherwise → `transformers.generate` with greedy decoding.

Override with `--inference {auto,vllm,hf}`.

Results are written to `eval_results/<ckpt-basename>/<benchmark>.json`.
Re-running skips benchmarks whose JSON already exists; pass `--force` to
overwrite.

### Eval JSON schema

Each `<benchmark>.json` is:

```jsonc
{
  "summary": {
    "score":     0.2667,                // aggregate metric (range depends on `metric`)
    "n":         300,                   // number of scored samples
    "metric":    "exact_number",        // 'accuracy' | 'exact_number' | benchmark-specific
    "n_prompts": 300,                   // prompts dispatched
    "elapsed_s": 18.4,                  // wall-clock for client.chat_batch
    "benchmark": "gsm8k",
    "ckpt_path": "/path/to/ckpt"
  },
  "samples": [
    {
      "pred_raw":  "Working it out... #### 42",   // raw model response
      "pred":      "42",                          // normalized prediction (string)
      "gold":      42.0,                          // gold value (type varies by bench)
      "correct":   1.0,                           // 0.0 or 1.0 (or in [0,1] for partial-credit benchmarks like ifeval)
      "meta":      {                              // benchmark-specific extras
        "subject": "...",                         //   mmlu/math_500
        "category":"...",                         //   truthful_qa
        "activity_label":"...",                   //   hellaswag
        "id":      "...",                         //   arc_challenge
        "question_id":"..."                       //   trivia_qa
      }
    },
    ...
  ]
}
```

`metric` legend: `accuracy` for letter-match benchmarks (mmlu, arc, hellaswag,
truthful_qa, mnli, ifeval, math_500, trivia_qa); `exact_number` for gsm8k.
For ifeval, `meta` also contains the four official sub-metrics
(`prompt_level_strict_acc`, `prompt_level_loose_acc`,
`inst_level_strict_acc`, `inst_level_loose_acc`); the headline `score` is
their unweighted mean.

## Tests

```bash
pytest                              # fast unit tests (~30 s, no network)
pytest -m slow                      # integration test, downloads SmolLM2-135M (~5 min)
```

## License

Apache-2.0. See `LICENSE`.

## Citation

```
@inproceedings{anonymous2026dghard,
  title  = {Spectral Unforgetting: Post-Hoc Recovery of Damaged Capabilities Without Retraining},
  author = {Anonymous},
  booktitle = {NeurIPS},
  year   = {2026},
}
```
