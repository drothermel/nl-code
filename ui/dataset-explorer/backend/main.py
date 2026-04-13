import json
import os
from pathlib import Path

# Force a non-interactive matplotlib backend before any request-time dataset
# validation code imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.routers.datasets import router as datasets_router

DEFAULT_ALLOW_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def _parse_allow_origins(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return DEFAULT_ALLOW_ORIGINS

    normalized_value = value.strip()
    if normalized_value.startswith("["):
        try:
            parsed_value = json.loads(normalized_value)
        except json.JSONDecodeError:
            parsed_value = None
        else:
            if isinstance(parsed_value, list):
                origins = [str(origin).strip() for origin in parsed_value if str(origin).strip()]
                if origins:
                    return origins

    origins = [origin.strip() for origin in normalized_value.split(",") if origin.strip()]
    return origins or DEFAULT_ALLOW_ORIGINS


allow_origins = _parse_allow_origins(
    os.getenv("CORS_ALLOW_ORIGINS") or os.getenv("ALLOWED_ORIGINS")
)

app = FastAPI(title="Dataset Explorer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router, prefix="/api/datasets", tags=["datasets"])


@app.get("/health")
def health():
    return {"status": "ok", "app": "dataset-explorer"}


FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
