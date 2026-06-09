# Testing

Use the checks below when asked to run live tests for this branch.

## Standard checks

```bash
uv run ruff check .
uv run ty check
uv run pytest
uv run nl-code-test docker -q
```

For the dataset explorer frontend:

```bash
cd ui/dataset-explorer/frontend
npm run check
```

## DSPy live smoke tests

These commands require `OPENROUTER_API_KEY` and a working Docker daemon. They
intentionally use tiny datasets and write outputs to `/tmp` so they only verify
that the live evaluation and optimizer jobs still start and complete.

By default, DSPy scripts resolve `DEFAULT_LLM_CONFIG_ID`
(`openrouter/xiaomi/mimo-v2-flash/off/v1`) through the OpenRouter catalog.
Pin `--llm-config-id` only when you need a different verified model.

If a live run fails before any local evaluation starts with an OpenRouter
"not a valid model ID" error, the default catalog entry or pinned
`--llm-config-id` has likely drifted from provider availability. Prefer fixing
the default catalog entry in `src/nl_code/optim/dspy_generators.py` instead of
only working around the issue at the command line.

Expect verbose DSPy output. `--auto light` is the smallest MIPRO budget exposed
by these scripts, but it still proposes several instruction candidates and runs
about 9-10 trials even with one train/dev/eval task. Warnings from DSPy about
ignored proposer fields are expected during these optimizer smoke tests; the
success signal is that the command writes an optimized program, summary, run
log, and event log under the requested `/tmp` output directory.

Run HumanEval DSPy eval with five samples:

Expected time: about 1 minute.

```bash
uv run python scripts/humaneval_dspy_eval.py \
  --generation-type both \
  --n-samples 5 \
  --num-repeats 1 \
  --output-dir /tmp/nl-code-live-dspy-eval
```

Run both MIPRO optimizer smoke tests with the smallest exposed optimizer budget:
`--auto light`. Keep one task in each required split and one worker thread.

Direct optimizer expected time: about 1-2 minutes.

```bash
uv run python scripts/optimize_humaneval_dspy_direct.py \
  --train-task-ids HumanEval/0 \
  --dev-task-ids HumanEval/1 \
  --eval-task-ids HumanEval/2 \
  --auto light \
  --num-threads 1 \
  --output-dir /tmp/nl-code-live-dspy-optimize-direct
```

Encoder-decoder optimizer expected time: about 2-3 minutes.

```bash
uv run python scripts/optimize_humaneval_dspy_encdec.py \
  --optimize-target both \
  --train-task-ids HumanEval/0 \
  --dev-task-ids HumanEval/1 \
  --eval-task-ids HumanEval/2 \
  --auto light \
  --num-threads 1 \
  --output-dir /tmp/nl-code-live-dspy-optimize-encdec
```

For the GEPA optimizer variants, the smallest smoke-test budget is
`--max-metric-calls 1`, which overrides `--auto`.
