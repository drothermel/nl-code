# nl-code

Primitives for research into LLMs and code generation. Provides dataset loading, code execution (with Docker isolation), code analysis, and a dataset explorer UI.

## Install

```bash
uv add nl-code                # core
uv add nl-code[docker]        # + Docker execution via dr-docker
uv add nl-code[bigcodebench]  # + scientific libs for BigCodeBench/ClassEval
```

## Code Execution

Execute generated code in isolated Docker containers (default) or local subprocesses.

Three execution modes covering all supported dataset test formats:

- **function_call** — call a named function with inputs, compare return values (HumanEval)
- **assertion** — exec code + assertion-based test code (HumanEval-Pro, MBPP-Pro, BigCodeBench Lite Pro)
- **unittest** — exec code + unittest.TestCase classes (ClassEval)

Batch variants (`batch_run_test_cases`, `batch_run_assertion_tests`, `batch_run_unittest_tests`) process many code samples in a single container with auto-chunking.

### Docker images

Build from the repo root:

```bash
docker build -t nl-code/code-eval:v1 -f docker/slim.Dockerfile .
docker build -t nl-code/code-eval-scientific:v1 -f docker/scientific.Dockerfile .
```

## Datasets

Loaders for HumanEval, HumanEval-Pro, MBPP-Pro, BigCodeBench Lite Pro, and ClassEval. Datasets are fetched from HuggingFace, parsed into `Task` objects, and cached locally. `DatasetSlice` supports filtering, seeded shuffling, and limit.

## Dataset Explorer

A FastAPI + React app for browsing and comparing datasets. Run from `ui/dataset-explorer/`.

## Headless validation runs

Some dataset validation tasks import `matplotlib`. Suppress GUI windows with:

```bash
MPLBACKEND=Agg uv run python ...
```
