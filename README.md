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

The corresponding raw task models preserve the original dataset inputs as `source__...` fields and expose richer derived artifacts such as:
- official prompt fields
- stripped and comment-preserving code stubs
- stripped and comment-preserving ground-truth code

Across task families, `new_official_prompt`, `new_code_stub`, and `new_code_stub_with_comments` provide a consistent interface for prompt/stub access even when the underlying dataset-specific field names differ.

`DatasetSlice` supports filtering, seeded shuffling, limits, and parallel accessors for common raw-task artifacts:
- `get_source_code(task_id)`
- `get_official_prompt(task_id)`
- `get_code_stub(task_id)`
- `get_code_stub_with_comments(task_id)`

## Dataset Explorer

A FastAPI + React app for browsing and comparing datasets. Run from `ui/dataset-explorer/`.

## HumanEval DSPy Experiments

This branch adds a small DSPy evaluation workflow for comparing direct code
generation against an encoder-decoder setup on HumanEval.

- `scripts/humaneval_dspy_eval.py` runs the evaluation from the command line.
  It writes a run JSON plus generation-history JSONL records under `logs/`.
- `src/nl_code/optim/humaneval_dspy_eval.py` contains the reusable evaluation
  loop, generation config, per-attempt results, and summary models.
- `src/nl_code/optim/dspy_generators.py` defines the direct generator and the
  encoder-decoder generator used by the eval.
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
