# AGENTS.md

## JSON I/O

Use Python's standard-library `json` module for JSON and JSONL I/O in this
repo. Do not add or use `srsly`; the DSPy/HumanEval artifacts can contain very
large integers that `srsly`'s `ujson` backend cannot reliably parse or
serialize.

## Pre-commit checks

Run all relevant checks before committing changes.

### Python (from repo root)

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
uv run nl-code-test docker
```

`uv run pytest` runs the default non-Docker suite. Run `uv run nl-code-test docker`
separately to execute the `@pytest.mark.docker` integration tests.

### Frontend (from ui/dataset-explorer/frontend/)

```bash
npx biome check --fix src/
npx tsc --noEmit
```

## Dataset validation

For dataset validation, debugging, or analysis commands that may import
`matplotlib`, run them headlessly with `MPLBACKEND=Agg` so they do not open GUI
plot windows during execution.
