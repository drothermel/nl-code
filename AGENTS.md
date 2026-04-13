# AGENTS.md

## Pre-commit checks

Run all relevant checks before committing changes.

### Python (from repo root)

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

### Frontend (from ui/dataset-explorer/frontend/)

```bash
npx biome check --fix src/
npx tsc --noEmit
```

## Dataset validation

For dataset validation, debugging, or analysis commands that may import
`matplotlib`, run them headlessly with `MPLBACKEND=Agg` so they do not open GUI
plot windows during execution.
