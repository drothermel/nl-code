import gzip
import json
import os
import shutil

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from nl_code.datasets.task import CodeDataset

PARSED_DATASET_CACHE_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
PAYLOAD_FILENAME = "payload.json.gz"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "nl-code" / "datasets"


class ParsedDatasetManifest(BaseModel):
    dataset_id: CodeDataset
    split: str
    source_revision: str | None = None
    cache_schema_version: int = PARSED_DATASET_CACHE_VERSION
    built_at: str
    raw_sample_count: int
    flawed_count: int
    task_count: int


class ParsedDatasetSnapshot(BaseModel):
    manifest: ParsedDatasetManifest
    raw_samples: dict[str, dict[str, Any]]
    flawed_raw_samples: dict[str, dict[str, Any]]
    tasks: dict[str, dict[str, Any]]


def get_cache_root() -> Path:
    env_value = os.environ.get("NL_CODE_DATASET_CACHE_DIR")
    if env_value:
        return Path(env_value).expanduser()
    return Path(DEFAULT_CACHE_DIR)


def get_cache_dir(dataset_id: CodeDataset | str, split: str) -> Path:
    safe_dataset_id = str(dataset_id).replace("/", "__")
    safe_split = split.replace("/", "__")
    return (
        get_cache_root()
        / safe_dataset_id
        / safe_split
        / f"v{PARSED_DATASET_CACHE_VERSION}"
    )


def read_manifest(
    dataset_id: CodeDataset | str, split: str
) -> ParsedDatasetManifest | None:
    manifest_path = get_cache_dir(dataset_id, split) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return ParsedDatasetManifest.model_validate_json(manifest_path.read_text())


def read_snapshot(
    dataset_id: CodeDataset | str, split: str
) -> ParsedDatasetSnapshot | None:
    cache_dir = get_cache_dir(dataset_id, split)
    manifest_path = cache_dir / MANIFEST_FILENAME
    payload_path = cache_dir / PAYLOAD_FILENAME
    if not manifest_path.exists() or not payload_path.exists():
        return None

    manifest = ParsedDatasetManifest.model_validate_json(manifest_path.read_text())
    if manifest.cache_schema_version != PARSED_DATASET_CACHE_VERSION:
        return None

    with gzip.open(payload_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    return ParsedDatasetSnapshot(
        manifest=manifest,
        raw_samples=payload["raw_samples"],
        flawed_raw_samples=payload["flawed_raw_samples"],
        tasks=payload["tasks"],
    )


def write_snapshot(snapshot: ParsedDatasetSnapshot) -> Path:
    cache_dir = get_cache_dir(snapshot.manifest.dataset_id, snapshot.manifest.split)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = cache_dir.parent / f"{cache_dir.name}.tmp-{uuid4().hex}"
    backup_dir = cache_dir.parent / f"{cache_dir.name}.bak-{uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        (staging_dir / MANIFEST_FILENAME).write_text(
            snapshot.manifest.model_dump_json(indent=2)
        )
        with gzip.open(
            staging_dir / PAYLOAD_FILENAME, "wt", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "raw_samples": snapshot.raw_samples,
                    "flawed_raw_samples": snapshot.flawed_raw_samples,
                    "tasks": snapshot.tasks,
                },
                handle,
            )

        backup_created = False
        if cache_dir.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            cache_dir.replace(backup_dir)
            backup_created = True

        try:
            staging_dir.replace(cache_dir)
        except Exception:
            if backup_created and backup_dir.exists() and not cache_dir.exists():
                backup_dir.replace(cache_dir)
            raise

        if backup_created and backup_dir.exists():
            shutil.rmtree(backup_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

    return cache_dir


def clear_snapshot(dataset_id: CodeDataset | str, split: str) -> bool:
    cache_dir = get_cache_dir(dataset_id, split)
    if not cache_dir.exists():
        return False
    shutil.rmtree(cache_dir)
    return True


def build_manifest(
    *,
    dataset_id: CodeDataset,
    split: str,
    source_revision: str | None,
    raw_sample_count: int,
    flawed_count: int,
    task_count: int,
) -> ParsedDatasetManifest:
    return ParsedDatasetManifest(
        dataset_id=dataset_id,
        split=split,
        source_revision=source_revision,
        built_at=datetime.now(UTC).isoformat(),
        raw_sample_count=raw_sample_count,
        flawed_count=flawed_count,
        task_count=task_count,
    )
