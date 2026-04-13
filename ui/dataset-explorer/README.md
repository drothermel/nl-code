# Dataset Explorer

Cross-dataset debugging and visualization app for the datasets implemented in
`nl_code.datasets`.

## Backend

From the repo root:

```bash
MPLBACKEND=Agg uv run uvicorn backend.main:app --app-dir ui/dataset-explorer --reload --port 8000
```

The backend also forces `matplotlib` onto the non-interactive `Agg` backend at
startup so dataset validation code that calls `plt.show()` does not open GUI
windows.

Set `CORS_ALLOW_ORIGINS` to a comma-separated string or JSON list to override
the default localhost CORS settings. `ALLOWED_ORIGINS` is also accepted as a
backward-compatible alias.

If a dataset was loaded before code or environment changes, refresh the cached
copy without restarting the server:

```bash
curl -X POST http://127.0.0.1:8000/api/datasets/bigcodebench-lite-pro/refresh
```

## Frontend

```bash
cd ui/dataset-explorer/frontend
npm ci
npm run dev
```

The Vite app proxies `/api` to the backend on port `8000`.

## Production-style local run

```bash
cd ui/dataset-explorer/frontend
npm ci
npm run build

# From repo root:
MPLBACKEND=Agg uv run uvicorn backend.main:app --app-dir ui/dataset-explorer --port 5000
```
