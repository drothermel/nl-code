# nl-code

Primitives for research into LLMs and code generation. Provides dataset loading, code execution (with Docker isolation), code analysis, and a dataset explorer UI.

## Install

```bash
uv add nl-code                # core
uv add nl-code[docker]        # + Docker execution via dr-docker
uv add nl-code[bigcodebench]  # + scientific libs for BigCodeBench/ClassEval
```

## Code Execution

Execute generated code in isolated Docker containers.

Three execution modes covering all supported dataset test formats:

- **function_call** — call a named function with inputs, compare return values (HumanEval)
- **assertion** — exec code + assertion-based test code (HumanEval-Pro, MBPP-Pro, BigCodeBench Lite Pro)
- **unittest** — exec code + unittest.TestCase classes (ClassEval)

Batch variants (`batch_run_test_cases`, `batch_run_assertion_tests`, `batch_run_unittest_tests`) process many code samples in a single container with auto-chunking.

### Build The Docker Image

Build the execution image from the repo root:

```bash
docker build -t nl-code/code-eval-scientific:v1 -f docker/scientific.Dockerfile .
```

This is the default runtime image used by the execution pipeline. The Dockerfile
installs both the `bigcodebench` dependency set and the pinned `dr-docker`
runtime dependency directly from `pyproject.toml`, so the image stays aligned
with the repo's declared execution requirements.

### Run The Docker Test Tier

Docker-dependent tests are marked with `@pytest.mark.docker` and are excluded
from the default `pytest` run.

Run them explicitly with:

```bash
uv run nl-code-test docker
```

You can pass extra pytest arguments through after `docker`, for example:

```bash
uv run nl-code-test docker -q tests/test_execution_runner.py
```

## Datasets

Loaders for HumanEval, HumanEval-Pro, MBPP-Pro, BigCodeBench Lite Pro, and ClassEval. Datasets are fetched from HuggingFace, parsed into `Task` objects, and cached locally.

Derived `Task` objects use schema version `v3`:
- `target: TaskTarget` with `name` and `kind` (`"function"` or `"class"`)
- `source: TaskSource` with runnable ground-truth code in `source.code`

Raw task models preserve the original dataset inputs in a nested `source` object. Derived artifacts such as ground-truth code, parsed test suites, and official prompts are exposed as `@cached_property` helpers (`gt_solution`, `test_suite`, `prompts`, and family-specific views) and are not serialized into cache payloads.

`DatasetSlice` supports filtering, seeded shuffling, limits, and accessors for common artifacts:
- `get_source_code(task_id)` — normalized runnable code from the derived `Task`
- `get_official_prompt(task_id)` — dataset-specific official prompt (HumanEval returns the raw HuggingFace prompt)

Parsed dataset caches use schema version 3. Rebuild after upgrading:

```bash
uv run python -m nl_code.datasets.cache_cli rebuild all
```

## Dataset Explorer

A FastAPI + React app for browsing and comparing datasets. Run from `ui/dataset-explorer/`.

## HumanEval DSPy Experiments

This branch adds a small DSPy evaluation workflow for comparing direct code
generation against an encoder-decoder setup on HumanEval.

- `scripts/humaneval_dspy_eval.py` runs the evaluation from the command line.
  It writes a run JSON plus generation-history JSONL records under `logs/`.
  ENCDEC eval defaults to stub encoder input (`raw.source.prompt`); pass
  `--encoder-input oracle` to feed `raw.gt_solution.code` for oracle round-trip checks.
- `scripts/optimize_humaneval_dspy_direct.py` and
  `scripts/optimize_humaneval_dspy_encdec.py` run MIPRO optimization for the
  direct and encoder-decoder HumanEval programs.
- `scripts/optimize_humaneval_dspy_direct_gepa.py` and
  `scripts/optimize_humaneval_dspy_encdec_gepa.py` run GEPA optimization for
  the same program families.
- `src/nl_code/optim/humaneval_dspy_eval.py` contains the reusable evaluation
  loop, generation config, per-attempt results, and summary models.
- `src/nl_code/optim/dspy_generators.py` defines the direct generator and the
  encoder-decoder generator used by the eval.
- `src/nl_code/optim/humaneval_dspy_optimize.py` and
  `src/nl_code/optim/humaneval_dspy_gepa.py` contain reusable optimizer
  orchestration, split handling, artifact writing, and summary models.
  Optimization event logging uses a per-context logger; `dspy.configure(lm=...)`
  remains process-global, so run one optimization or eval job per process.
- `src/nl_code/optim/humaneval_dspy_logs.py` parses eval logs into a nested
  Pydantic snapshot for notebook analysis. It preserves run stats, per-attempt
  results, and individual LM calls, including both encoder and decoder calls
  for new encoder-decoder runs.
- `scripts/parse_humaneval_dspy_logs.py` is a thin wrapper that parses the
  current `logs/` directory into a snapshot JSON.
- `nbs/exp/human_eval_dspy.py` is a marimo notebook for inspecting the workflow,
  loading the parsed snapshot, comparing pass rates, and stepping through failed
  cases side by side for direct and encoder-decoder generations.
- `scripts/sample_humaneval_dspy_splits.py` samples train/dev/eval task splits
  from the full direct and encoder-decoder eval logs.

Typical usage:

```bash
OPENROUTER_API_KEY=... uv run python scripts/humaneval_dspy_eval.py --generation-type both --n-samples 20
uv run python scripts/parse_humaneval_dspy_logs.py --logs-dir logs --output-path logs/human_eval_dspy_snapshot_latest.json
uv run marimo edit nbs/exp/human_eval_dspy.py
```

### DSPy Log And Report Inspection

Forensic tooling works in layers. Flat `logs/` output from eval and optimization
is not a session root on its own.

```text
logs/  ──parse_humaneval_dspy_logs.py──►  one aggregate snapshot JSON
logs/  ──sessionize_dspy_logs_v0.py────►  sessionized corpus (metadata.json + raw/)
sessionized corpus  ──inspect_dspy_* --walk──►  parsed_*_reports/
parsed_gepa_reports/  ──build_dspy_gepa_agent_bundle.py──►  agent bundle JSON
```

Use `parse_humaneval_dspy_logs.py` for quick notebook-style exploration across
all files in `logs/`. Use `sessionize_dspy_logs_v0.py` before
`inspect_dspy_eval_session.py` or `inspect_dspy_gepa_session.py`. Those inspect
scripts require a session directory containing `metadata.json`; pointing them at
raw subdirectories such as `logs/eval_full_5x/baseline_direct` will fail.

The canonical sessionized corpus lives outside the repo at
`~/drotherm/data/code-comp/dspy-exps/v0`. Regenerate it from the repo root:

```bash
SESSIONIZE_SOURCE_ROOT=$PWD \
SESSIONIZE_OUTPUT_ROOT=~/drotherm/data/code-comp/dspy-exps/v0 \
uv run python scripts/sessionize_dspy_logs_v0.py

uv run python scripts/inspect_dspy_eval_session.py \
  ~/drotherm/data/code-comp/dspy-exps/v0 --walk

uv run python scripts/inspect_dspy_gepa_session.py \
  ~/drotherm/data/code-comp/dspy-exps/v0 --walk

uv run python scripts/build_dspy_gepa_agent_bundle.py \
  ~/drotherm/data/code-comp/dspy-exps/v0/parsed_gepa_reports
```

Scripts:

- `scripts/sessionize_dspy_logs_v0.py` groups raw DSPy log artifacts into
  session directories and writes session metadata.
- `scripts/inspect_dspy_eval_session.py` parses one eval session, or walks a
  corpus, into `*.eval_report.json` files with runs, samples, attempts,
  generation calls, aggregates, and parse notes.
- `scripts/inspect_dspy_gepa_session.py` parses one GEPA optimizer session, or
  walks a corpus, into `*.gepa_report.json` files with optimizer runs, programs,
  split/task scores, metric calls, generated outputs, optimizer iterations, and
  safe `gepa_state.bin` metadata scans.
- `scripts/build_dspy_gepa_agent_bundle.py` combines the per-session GEPA
  reports into one cross-session `gepa_optimization_agent_bundle.json` for
  downstream analysis agents or UI tooling. The bundle omits raw LLM request
  payloads; treat parsed forensic reports as sensitive if shared externally.
- `docs/dspy-log-sessions-v0.md` documents the sessionized log corpus and
  sessionization rules.
- `docs/dspy-eval-optimizer-extraction-progress.md` records extraction progress
  and the known limits of eval versus optimizer logs.
- `docs/session_000018_gepa_prompt_variants.md` is a concrete session-level
  prompt-variant review for the most complete direct GEPA trace.

The report extractors use Python's standard-library `json` module because these
artifacts can contain very large integers that are not safe with `srsly`'s
`ujson` backend.

### DSPy Static Viewer

`ui/dspy-eval-static-viewer/` contains a self-contained static viewer generated
from the parsed eval and GEPA reports. Open
`ui/dspy-eval-static-viewer/viewer.html` directly in a browser; it loads
`data/viewer_data.js` locally and does not require a backend server.

The viewer includes:

- a GEPA prompt-flow tab with full prompt text, candidate lineage, scores, and
  per-task heatmaps;
- a HumanEval full-5x sample variation matrix with task drilldowns; and
- CSV exports for prompt nodes and stable/unstable task summaries.

The committed viewer is isolated from the existing `ui/dataset-explorer` app.
It intentionally includes only the browser-loadable data bundle and CSV exports,
not the duplicate JSON payload or one-off preprocessing script from the original
Desktop bundle.

## Headless validation runs

General dataset validation/debugging commands that import `matplotlib` should run headlessly with:

```bash
MPLBACKEND=Agg uv run python ...
```

## Rebuild Dataset Caches

Run the Docker-backed cache rebuilds with:

```bash
uv run python -m nl_code.datasets.cache_cli rebuild all
uv run python -m nl_code.datasets.cache_cli rebuild humaneval-plus
uv run python -m nl_code.datasets.cache_cli rebuild humaneval-pro
uv run python -m nl_code.datasets.cache_cli rebuild mbpp-pro
uv run python -m nl_code.datasets.cache_cli rebuild class-eval
uv run python -m nl_code.datasets.cache_cli rebuild bigcodebench-lite-pro
```

`cache_cli rebuild` sets `MPLBACKEND=Agg` automatically.

Current observed results with the default execution image and env limits:

```text
humaneval-plus: cached 163 tasks (163 raw, 1 flawed)
humaneval-pro: cached 163 tasks (163 raw, 1 flawed)
mbpp-pro: cached 375 tasks (375 raw, 3 flawed)
class-eval: cached 98 tasks (98 raw, 2 flawed)
bigcodebench-lite-pro: cached 54 tasks (54 raw, 3 flawed)
```

The remaining flawed samples above are dataset-level failures, not Docker
runtime failures.

The current known flawed HumanEval-Pro sample is `HumanEvalPro/24`, where the
new function docstring is not present in `new_solution`.
