# CLAUDE.md

## Pre-commit checks

Run these before committing:

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

## Dataset validation

For dataset validation, debugging, or analysis commands that may import
`matplotlib`, run them with `MPLBACKEND=Agg` to suppress GUI plot windows:

```bash
MPLBACKEND=Agg uv run python ...
```
