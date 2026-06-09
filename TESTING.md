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
Pin `--llm-config-id openrouter/xiaomi/mimo-v2-flash/off/v1` so the smoke tests
do not depend on a stale or unavailable default model.

Run HumanEval DSPy eval with five samples:

```bash
uv run python scripts/humaneval_dspy_eval.py \
  --generation-type both \
  --n-samples 5 \
  --num-repeats 1 \
  --llm-config-id openrouter/xiaomi/mimo-v2-flash/off/v1 \
  --output-dir /tmp/nl-code-live-dspy-eval
```

Run both MIPRO optimizer smoke tests with the smallest exposed optimizer budget:
`--auto light`. Keep one task in each required split and one worker thread.

```bash
uv run python scripts/optimize_humaneval_dspy_direct.py \
  --train-task-ids HumanEval/0 \
  --dev-task-ids HumanEval/1 \
  --eval-task-ids HumanEval/2 \
  --llm-config-id openrouter/xiaomi/mimo-v2-flash/off/v1 \
  --auto light \
  --num-threads 1 \
  --output-dir /tmp/nl-code-live-dspy-optimize-direct
```

```bash
uv run python scripts/optimize_humaneval_dspy_encdec.py \
  --optimize-target both \
  --train-task-ids HumanEval/0 \
  --dev-task-ids HumanEval/1 \
  --eval-task-ids HumanEval/2 \
  --llm-config-id openrouter/xiaomi/mimo-v2-flash/off/v1 \
  --auto light \
  --num-threads 1 \
  --output-dir /tmp/nl-code-live-dspy-optimize-encdec
```

For the GEPA optimizer variants, the smallest smoke-test budget is
`--max-metric-calls 1`, which overrides `--auto`.
