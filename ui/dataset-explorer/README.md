# Dataset Explorer

Cross-dataset debugging and visualization app for the datasets implemented in
`nl_code.datasets`.

## Backend

From the repo root:

```bash
uv run uvicorn backend.main:app --app-dir ui/dataset-explorer --reload --port 8000
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
uv run uvicorn backend.main:app --app-dir ui/dataset-explorer --port 5000
```
