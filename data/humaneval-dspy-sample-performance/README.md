# HumanEval DSPy Sample Performance

This directory contains derived sample-level performance artifacts from the
sessionized DSPy HumanEval run corpus.

## Source Data

The artifacts are generated from the canonical sessionized DSPy corpus at:

```text
/Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0
```

The exporter reads:

- `parsed_eval_reports/*.eval_report.json` for eval attempts and per-attempt
  `test_pass_rate` values.
- `session_*/raw/logs/dspy_optimized/*_summary.json` for optimizer baseline
  and final optimized per-task scores.
- `parsed_gepa_reports/*.gepa_report.json` for GEPA candidate/search metric
  calls.
- `session_*/raw/logs/dspy_optimized/*_events.jsonl` for MIPRO candidate/search
  metric calls.

Each performance value is a pass-rate value already aggregated across the test
cases for a single HumanEval sample.

## Regeneration

From the repo root:

```bash
uv run python scripts/export_humaneval_sample_performance.py \
  --output-dir data/humaneval-dspy-sample-performance
```

The script uses the default canonical corpus path above unless `--corpus-root`
is provided.

## Files

- `humaneval_sample_performance_evidence.csv`
  - Long-form audit table with one row per observed sample-performance value.
  - Includes eval attempts, optimizer summary task scores, GEPA search metric
    calls, and MIPRO search metric calls.

- `humaneval_sample_performance_aggregate.csv`
  - Aggregates the evidence table by generation family (`direct` or `enc-dec`),
    sample ID, and model name.
  - Includes baseline, non-baseline, and all-evidence means, variances, and
    counts.

- `humaneval_gpt5nano_all_settings_sample_performance.csv`
  - One row per sample ID for `openrouter/openai/gpt-5-nano`.
  - Aggregates across direct, enc-dec, baseline, and non-baseline evidence.

- `humaneval_gpt5nano_direct_sample_performance.csv`
  - One row per sample ID for GPT-5 nano direct evidence only.

- `humaneval_gpt5nano_encdec_sample_performance.csv`
  - One row per sample ID for GPT-5 nano encoder-decoder evidence only.

- `humaneval_gpt5nano_baseline_sample_performance.csv`
  - One row per sample ID for GPT-5 nano baseline evidence only.

- `humaneval_gpt5nano_non_baseline_sample_performance.csv`
  - One row per sample ID for GPT-5 nano non-baseline evidence only.

- `humaneval_gpt5nano_worst_all_settings_sample_sets.json`
  - JSON selection artifact for dataset construction.
  - Contains worst-performing GPT-5 nano sample ID sets for `N = 25`, `50`, and
    `100`, selected by ascending all-settings average performance.
  - Also includes per-selected-sample mean, variance, and count statistics for
    `all`, `direct`, `enc-dec`, `baseline`, and `non-baseline` settings, plus
    the all-settings selection rank.
