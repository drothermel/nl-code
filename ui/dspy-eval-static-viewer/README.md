# DSPy Eval Static Viewer

This directory contains a self-contained static viewer for the parsed DSPy GEPA
and HumanEval evaluation outputs.

Open `viewer.html` directly in a browser. It loads `data/viewer_data.js` with a
local script tag, so no backend server is required.

## Included

```text
viewer.html
data/viewer_data.js
exports/gepa_prompt_nodes.csv
exports/stable_tasks_full5x.csv
exports/task_variation_full5x.csv
exports/unstable_tasks_full5x.csv
```

The original Desktop bundle also included `data/viewer_data.json`,
`preprocess_eval_reports.py`, `_inline_check.js`, and `__pycache__/`. Those are
not committed here:

- `viewer_data.json` duplicates the 44 MB JavaScript data payload.
- `preprocess_eval_reports.py` is the one-off generator from the external
  bundle, not repo-owned application code.
- `_inline_check.js` duplicates the viewer's inline script.
- `__pycache__/` is generated Python bytecode.

## Data Summary

The bundled data was validated locally after extraction:

```text
GEPA sessions:     9
GEPA prompt nodes: 128
GEPA prompt edges: 113
Full 5x tasks:     163
```
