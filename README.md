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

Loaders for HumanEval, HumanEval-Pro, MBPP-Pro, BigCodeBench Lite Pro, and ClassEval. Datasets are fetched from HuggingFace, parsed into `Task` objects, and cached locally. `DatasetSlice` supports filtering, seeded shuffling, and limit.

## Dataset Explorer

A FastAPI + React app for browsing and comparing datasets. Run from `ui/dataset-explorer/`.

## Headless validation runs

General dataset validation/debugging commands that import `matplotlib` should run headlessly with:

```bash
MPLBACKEND=Agg uv run python ...
```

## Rebuild Dataset Caches

Run the Docker-backed cache rebuilds with:

```bash
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
humaneval-pro: cached 164 tasks (164 raw, 0 flawed)
mbpp-pro: cached 375 tasks (375 raw, 3 flawed)
class-eval: cached 98 tasks (98 raw, 2 flawed)
bigcodebench-lite-pro: cached 54 tasks (54 raw, 3 flawed)
```

The remaining flawed samples above are dataset-level failures, not Docker
runtime failures.
