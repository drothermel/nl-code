# DSPy Eval and Optimizer Extraction Progress

This note tracks the follow-on extraction work for the normalized DSPy log
corpus documented in `docs/dspy-log-sessions-v0.md`.

The external corpus root is:

```text
/Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0
```

## Completed Work

Added an eval-session inspection and export script:

```text
scripts/inspect_dspy_eval_session.py
```

The script supports:

- single-session CLI inspection with Rich rendering
- `-n` sample limiting for terminal exploration
- `-o/--output-file` for writing one parsed eval report
- `--walk` mode for crawling the top-level `v0/` directory and writing one
  parsed report per eval session

The output format is forensic JSON with these top-level sections:

```text
session
runs
samples
attempts
generation_calls
aggregates
parse_notes
```

Each report preserves per-sample and per-attempt data where available, including
task ids, generated code, pass rates, testcase results, generation-call
metadata, prompt fingerprints, model metadata, and aggregate summaries.

## Generated Eval Reports

The eval extractor has been run across the `v0/` corpus:

```bash
uv run python scripts/inspect_dspy_eval_session.py /Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0 --walk
```

It wrote reports under:

```text
/Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0/parsed_eval_reports
```

The run produced 14 eval report JSON files plus a `manifest.json`. It skipped
19 non-eval sessions.

The full 5x HumanEval comparison is split across these eval sessions:

```text
session_000025  baseline direct
session_000026  GEPA direct
session_000027  GEPA encdec decoder
session_000028  MIPRO direct
session_000029  MIPRO encdec decoder
session_000030  GEPA encdec encoder
session_000031  MIPRO encdec encoder
session_000032  baseline encdec
```

## Baseline Eval Findings

The standalone baseline eval sessions are:

```text
session_000001  direct baseline, 20 samples, legacy eval
session_000002  encdec baseline, 20 samples, legacy eval
session_000003  direct baseline, 157 outputs from 163 selected, legacy eval
session_000004  encdec baseline, 157 outputs from 163 selected, legacy eval
session_000005  direct baseline, 69 samples x 5 repeats
session_000006  encdec baseline, 69 samples x 5 repeats
session_000025  direct baseline, 163 samples x 5 repeats
session_000032  encdec baseline, 163 samples x 5 repeats
```

For standalone eval-output sessions, the baseline model set is:

```text
openrouter/openai/gpt-5-nano
```

The observed response model is:

```text
openai/gpt-5-nano-2025-08-07
```

Optimizer-internal scoring also includes `openrouter/xiaomi/mimo-v2-flash`, but
that is not part of the standalone baseline eval-output set.

## Optimizer Findings

Only two optimizer families have been found in the `v0/` corpus:

```text
MIPROv2
GEPA
```

Optimizer summaries include per-task score maps for baseline and optimized
programs across `train`, `dev`, and `eval` splits. Those maps are useful for
coarse comparisons, but they are not equivalent to the eval reports because
they generally do not preserve full per-repeat attempts, generated code for
every scored task, full testcase result lists, or a reliable per-call mapping
from prompt to scored task.

MIPRO logs do not appear to have enough prompt/evaluation detail to be worth a
dedicated extraction pass for the current goal. They preserve optimized program
artifacts and summary score maps, but not enough recoverable candidate prompt
text paired with individual evaluation outcomes to justify a special MIPRO
extractor beyond the existing summary-level optimizer metadata.

GEPA is more promising for a dedicated pass. GEPA sessions include summary
files, optimized program files, event logs, run logs, generated-best-output
files, and `gepa_state.bin`. The event logs include `metric_call` records with
task ids and pass rates, and the generated-best-output files preserve generated
code for validation-set tasks. Further GEPA-specific inspection is still needed
to determine whether candidate/reflection text and prompt revisions can be
paired cleanly with those metric calls.

## Handoff Prompts Prepared

Prepared separate investigation prompts for other agents:

- MIPRO: inspect one representative MIPRO optimizer run and inventory any
  extractable `(prompt or program, evaluation result)` pairs.
- GEPA: inspect one representative GEPA optimizer run and inventory extractable
  prompt/program, metric-call, generated-output, and possible reflection data.

The recommended representative sessions are:

```text
session_000007  compact direct MIPRO run
session_000018  direct GEPA run with events, run log, generated outputs, and state
```

## Open Next Steps

The next useful implementation is likely a GEPA-specific optimizer extractor
that normalizes:

- session metadata
- optimized program metadata
- summary-level baseline and optimized task scores
- event-log metric calls
- generated-best-output files
- any recoverable GEPA prompt/reflection/candidate text

MIPRO should remain summary-level unless new evidence shows that its logs can
reliably connect candidate prompts to individual task scores.
