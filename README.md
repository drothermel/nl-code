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
installs the `bigcodebench` dependency set directly from `pyproject.toml`, so
the image stays aligned with the repo's declared scientific requirements.

## Datasets

Loaders for HumanEval, HumanEval-Pro, MBPP-Pro, BigCodeBench Lite Pro, and ClassEval. Datasets are fetched from HuggingFace, parsed into `Task` objects, and cached locally. `DatasetSlice` supports filtering, seeded shuffling, and limit.

## Dataset Explorer

A FastAPI + React app for browsing and comparing datasets. Run from `ui/dataset-explorer/`.

## Headless validation runs

Some dataset validation tasks import `matplotlib`. Suppress GUI windows with:

```bash
MPLBACKEND=Agg uv run python ...
```
