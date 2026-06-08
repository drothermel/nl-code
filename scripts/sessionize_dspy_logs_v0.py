from __future__ import annotations

import hashlib
import json as stdlib_json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import srsly
from pydantic import BaseModel, ConfigDict, Field


SOURCE_ROOT = Path("/Users/daniellerothermel/drotherm/repos/nl-code")
OUTPUT_ROOT = Path("~/drotherm/data/code-comp/dspy-exps/v0").expanduser()
OVERWRITE_OUTPUT = False
OVERWRITE_ENV_VAR = "SESSIONIZE_DSPY_LOGS_OVERWRITE"

SCHEMA_VERSION = "v0"
UNKNOWN_SESSION_ID = "unknown_session"
LOG_DIR_PATTERN = re.compile(r"log_dir=(\S+)")
LEGACY_EVAL_FILE_PATTERN = re.compile(
    r"^human_eval_dspy_(direct|encdec)_eval_\d{8}T\d{6}Z\.json$"
)


class FileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_path: str
    original_relative_path: str
    copied_path: str
    role: str
    extension: str
    size_bytes: int
    sha256: str
    record_count: int | None = None
    parse_error: str | None = None
    unknown_reason: str | None = None


class RelationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_type: str
    confidence: str
    source_file: str
    target_file: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SessionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    session_id: str
    session_kind: str
    confidence: str
    created_at: datetime
    source_roots: list[str]
    original_grouping_paths: list[str] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    files: list[FileMetadata] = Field(default_factory=list)
    relation_evidence: list[RelationEvidence] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)
    unknown_reasons: list[dict[str, str]] = Field(default_factory=list)


class PlannedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    role: str
    record_count: int | None = None
    parse_error: str | None = None
    unknown_reason: str | None = None


class PlannedSession(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_kind: str
    confidence: str
    primary_path: Path
    sort_timestamp: str | None = None
    original_grouping_paths: list[str] = Field(default_factory=list)
    extracted: dict[str, Any] = Field(default_factory=dict)
    files: list[PlannedFile] = Field(default_factory=list)
    relation_evidence: list[RelationEvidence] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    created_at: datetime
    source_root: str
    output_root: str
    session_count: int
    unknown_file_count: int
    total_discovered_files: int
    total_copied_files: int
    total_copied_bytes: int
    sessions: list[str]


def main() -> None:
    source_files = discover_source_files()
    sessions = build_sessions(source_files)
    assigned_paths = {
        planned_file.path for session in sessions for planned_file in session.files
    }
    unknown_files = [
        PlannedFile(
            path=path,
            role=role_for_unknown_file(path),
            record_count=record_count_for_path(path),
            unknown_reason=unknown_reason_for_path(path),
        )
        for path in source_files
        if path not in assigned_paths
    ]

    sessions = assign_session_ids(sessions)
    write_corpus(sessions, unknown_files, len(source_files))


def discover_source_files() -> list[Path]:
    roots = [SOURCE_ROOT / "logs", SOURCE_ROOT / "los"]
    return sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    )


def build_sessions(source_files: list[Path]) -> list[PlannedSession]:
    source_file_set = set(source_files)
    sessions: list[PlannedSession] = []
    consumed: set[Path] = set()

    for path in sorted(source_file_set):
        if path.name.startswith("human_eval_dspy_run_") and path.suffix == ".json":
            session = build_eval_run_session(path)
            sessions.append(session)
            consumed.update(planned_file.path for planned_file in session.files)

    for path in sorted(source_file_set - consumed):
        if is_legacy_eval_path(path):
            session = build_legacy_eval_session(path)
            sessions.append(session)
            consumed.update(planned_file.path for planned_file in session.files)

    for path in sorted(
        (SOURCE_ROOT / "logs" / "dspy_optimized").glob("*_summary.json")
    ):
        if path in consumed:
            continue
        session = build_optimization_session(path)
        sessions.append(session)
        consumed.update(planned_file.path for planned_file in session.files)

    return sessions


def build_eval_run_session(path: Path) -> PlannedSession:
    payload = read_json_file(path)
    config = dict(payload.get("config") or {})
    attempts = list(payload.get("attempts") or [])
    skipped_count = sum(1 for attempt in attempts if attempt.get("skipped") is True)
    non_skipped_count = len(attempts) - skipped_count

    generation_refs = sorted(
        {
            resolve_source_path(attempt.get("generation_log_file"))
            for attempt in attempts
            if attempt.get("generation_log_file")
        }
    )
    generation_refs = [ref for ref in generation_refs if ref.exists()]
    generation_row_counts = {
        relative_source_path(ref): count_jsonl_records(ref) for ref in generation_refs
    }
    validation_notes = generation_count_notes(non_skipped_count, generation_row_counts)

    files = [PlannedFile(path=path, role="run_result")]
    files.extend(
        PlannedFile(
            path=ref,
            role="generation_history",
            record_count=generation_row_counts[relative_source_path(ref)],
        )
        for ref in generation_refs
    )

    relation_evidence = [
        RelationEvidence(
            evidence_type="run_attempt_generation_log_file",
            confidence="exact",
            source_file=relative_source_path(path),
            target_file=relative_source_path(ref),
            details={
                "referencing_attempt_count": sum(
                    1
                    for attempt in attempts
                    if resolve_source_path(attempt.get("generation_log_file")) == ref
                ),
            },
        )
        for ref in generation_refs
    ]

    summary = dict(payload.get("summaries") or {})
    extracted = {
        "timestamp": payload.get("timestamp"),
        "generation_type": config.get("generation_type"),
        "model": config.get("model"),
        "seed": config.get("seed"),
        "num_repeats": config.get("num_repeats"),
        "attempt_count": len(attempts),
        "skipped_count": skipped_count,
        "non_skipped_count": non_skipped_count,
        "generation_call_count": sum(generation_row_counts.values()),
        "selected_dataset_indices_count": len(
            payload.get("selected_dataset_indices") or []
        ),
        "metrics": summary,
        "config": config,
    }

    return PlannedSession(
        session_kind="eval_run",
        confidence="exact" if generation_refs else "medium",
        primary_path=path,
        sort_timestamp=payload.get("timestamp"),
        original_grouping_paths=[relative_source_path(path.parent)],
        extracted=extracted,
        files=dedupe_planned_files(files),
        relation_evidence=relation_evidence,
        validation_notes=validation_notes,
    )


def build_legacy_eval_session(path: Path) -> PlannedSession:
    payload = read_json_file(path)
    outputs = list(payload.get("outputs") or [])
    dataset_indices = list(payload.get("dataset_indices") or [])
    extracted = {
        "timestamp": payload.get("timestamp"),
        "eval_type": payload.get("eval_type"),
        "generation_type": payload.get("eval_type"),
        "seed": payload.get("seed"),
        "val_num": payload.get("val_num"),
        "dataset_indices_count": len(dataset_indices),
        "outputs_count": len(outputs),
    }
    return PlannedSession(
        session_kind="legacy_eval",
        confidence="exact",
        primary_path=path,
        sort_timestamp=payload.get("timestamp"),
        original_grouping_paths=[relative_source_path(path.parent)],
        extracted=extracted,
        files=[PlannedFile(path=path, role="legacy_eval_result")],
        relation_evidence=[
            RelationEvidence(
                evidence_type="self_contained_legacy_eval_file",
                confidence="exact",
                source_file=relative_source_path(path),
            )
        ],
    )


def build_optimization_session(summary_path: Path) -> PlannedSession:
    summary = read_json_file(summary_path)
    candidate_paths = optimizer_candidate_paths(summary_path, summary)
    files = [
        PlannedFile(path=path, role=role_for_optimizer_file(path))
        for path in sorted(candidate_paths)
        if path.exists() and path.is_file()
    ]

    relation_evidence = [
        RelationEvidence(
            evidence_type="optimizer_summary_file_reference",
            confidence="exact",
            source_file=relative_source_path(summary_path),
            target_file=relative_source_path(path),
        )
        for path in sorted(candidate_paths)
        if path.exists() and path.is_file() and path != summary_path
    ]

    validation_notes: list[str] = []
    for run_log_path in run_log_paths_for_summary(summary):
        if not run_log_path.exists():
            continue
        for log_dir in optimizer_internal_log_dirs(run_log_path):
            if not log_dir.exists():
                validation_notes.append(
                    f"Referenced optimizer log dir does not exist: {relative_source_path(log_dir)}"
                )
                continue
            internal_files = sorted(
                path for path in log_dir.rglob("*") if path.is_file()
            )
            files.extend(
                PlannedFile(path=path, role=role_for_optimizer_internal_file(path))
                for path in internal_files
            )
            relation_evidence.append(
                RelationEvidence(
                    evidence_type="optimizer_run_log_dir_reference",
                    confidence="exact",
                    source_file=relative_source_path(run_log_path),
                    target_file=relative_source_path(log_dir),
                    details={"file_count": len(internal_files)},
                )
            )

    extracted = {
        "timestamp": summary.get("timestamp"),
        "generation_type": summary.get("generation_type"),
        "optimization_target": summary.get("optimization_target"),
        "model": summary.get("model"),
        "seed": summary.get("seed"),
        "auto": summary.get("auto"),
        "max_metric_calls": summary.get("max_metric_calls"),
        "num_threads": summary.get("num_threads"),
        "train_task_count": len(summary.get("train_task_ids") or []),
        "dev_task_count": len(summary.get("dev_task_ids") or []),
        "eval_task_count": len(summary.get("eval_task_ids") or []),
        "baseline_scores": summary.get("baseline_scores"),
        "optimized_scores": summary.get("optimized_scores"),
    }

    return PlannedSession(
        session_kind="optimization",
        confidence="exact",
        primary_path=summary_path,
        sort_timestamp=summary.get("timestamp"),
        original_grouping_paths=[relative_source_path(summary_path.parent)],
        extracted=extracted,
        files=dedupe_planned_files(files),
        relation_evidence=relation_evidence,
        validation_notes=validation_notes,
    )


def optimizer_candidate_paths(summary_path: Path, summary: dict[str, Any]) -> set[Path]:
    paths = {summary_path}
    for key in [
        "optimized_program_path",
        "summary_path",
        "event_log_path",
        "run_log_path",
    ]:
        value = summary.get(key)
        if value:
            paths.add(resolve_source_path(value))

    stem = summary_path.name.removesuffix("_summary.json")
    paths.update(summary_path.parent.glob(f"{stem}*"))
    return paths


def run_log_paths_for_summary(summary: dict[str, Any]) -> list[Path]:
    run_log_path = summary.get("run_log_path")
    if not run_log_path:
        return []
    return [resolve_source_path(run_log_path)]


def optimizer_internal_log_dirs(run_log_path: Path) -> list[Path]:
    text = run_log_path.read_text(encoding="utf-8", errors="replace")
    return sorted(
        {resolve_source_path(match) for match in LOG_DIR_PATTERN.findall(text)}
    )


def assign_session_ids(sessions: list[PlannedSession]) -> list[PlannedSession]:
    return sorted(
        sessions,
        key=lambda session: (
            session.sort_timestamp or "",
            relative_source_path(session.primary_path),
        ),
    )


def write_corpus(
    sessions: list[PlannedSession],
    unknown_files: list[PlannedFile],
    total_discovered_files: int,
) -> None:
    prepare_output_root()
    created_at = datetime.now(timezone.utc)
    copied_files: list[FileMetadata] = []
    session_ids: list[str] = []

    for index, session in enumerate(sessions, start=1):
        session_id = f"session_{index:06d}"
        session_ids.append(session_id)
        session_dir = OUTPUT_ROOT / session_id
        file_metadata = copy_session_files(session_dir, session.files)
        copied_files.extend(file_metadata)
        metadata = SessionMetadata(
            session_id=session_id,
            session_kind=session.session_kind,
            confidence=session.confidence,
            created_at=created_at,
            source_roots=[str(SOURCE_ROOT)],
            original_grouping_paths=session.original_grouping_paths,
            extracted=session.extracted,
            files=file_metadata,
            relation_evidence=session.relation_evidence,
            validation_notes=session.validation_notes,
        )
        write_metadata(session_dir / "metadata.json", metadata)

    unknown_metadata_files = write_unknown_session(unknown_files, created_at)
    copied_files.extend(unknown_metadata_files)

    manifest = Manifest(
        created_at=created_at,
        source_root=str(SOURCE_ROOT),
        output_root=str(OUTPUT_ROOT),
        session_count=len(sessions),
        unknown_file_count=len(unknown_files),
        total_discovered_files=total_discovered_files,
        total_copied_files=len(copied_files),
        total_copied_bytes=sum(file.size_bytes for file in copied_files),
        sessions=session_ids + [UNKNOWN_SESSION_ID],
    )
    write_json_file(OUTPUT_ROOT / "manifest.json", manifest.model_dump(mode="json"))

    print(f"Output: {OUTPUT_ROOT}")
    print(f"Sessions: {len(sessions)}")
    print(f"Unknown files: {len(unknown_files)}")
    print(f"Copied files: {len(copied_files)}")


def write_unknown_session(
    unknown_files: list[PlannedFile],
    created_at: datetime,
) -> list[FileMetadata]:
    session_dir = OUTPUT_ROOT / UNKNOWN_SESSION_ID
    file_metadata = copy_session_files(session_dir, unknown_files)
    metadata = SessionMetadata(
        session_id=UNKNOWN_SESSION_ID,
        session_kind="unknown_session",
        confidence="unknown",
        created_at=created_at,
        source_roots=[str(SOURCE_ROOT)],
        files=file_metadata,
        unknown_reasons=[
            {
                "original_relative_path": file.original_relative_path,
                "reason": file.unknown_reason or "unassigned",
            }
            for file in file_metadata
        ],
    )
    write_metadata(session_dir / "metadata.json", metadata)
    return file_metadata


def copy_session_files(
    session_dir: Path, planned_files: list[PlannedFile]
) -> list[FileMetadata]:
    metadata: list[FileMetadata] = []
    for planned_file in dedupe_planned_files(planned_files):
        source_path = planned_file.path
        copied_path = raw_copy_path(session_dir, source_path)
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, copied_path)
        metadata.append(file_metadata_for_path(session_dir, planned_file, copied_path))
    return metadata


def prepare_output_root() -> None:
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(parents=True)
        return
    if OVERWRITE_OUTPUT or os.environ.get(OVERWRITE_ENV_VAR) == "1":
        shutil.rmtree(OUTPUT_ROOT)
        OUTPUT_ROOT.mkdir(parents=True)
        return
    if any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_ROOT} already exists and is not empty. "
            "Set OVERWRITE_OUTPUT = True to replace it."
        )


def file_metadata_for_path(
    session_dir: Path,
    planned_file: PlannedFile,
    copied_path: Path,
) -> FileMetadata:
    source_path = planned_file.path
    return FileMetadata(
        original_path=str(source_path),
        original_relative_path=relative_source_path(source_path),
        copied_path=str(copied_path.relative_to(session_dir)),
        role=planned_file.role,
        extension=source_path.suffix or "[no extension]",
        size_bytes=source_path.stat().st_size,
        sha256=sha256_file(source_path),
        record_count=planned_file.record_count,
        parse_error=planned_file.parse_error,
        unknown_reason=planned_file.unknown_reason,
    )


def raw_copy_path(session_dir: Path, source_path: Path) -> Path:
    return session_dir / "raw" / source_path.relative_to(SOURCE_ROOT)


def dedupe_planned_files(planned_files: list[PlannedFile]) -> list[PlannedFile]:
    seen: set[Path] = set()
    result: list[PlannedFile] = []
    for planned_file in planned_files:
        if planned_file.path in seen:
            continue
        seen.add(planned_file.path)
        result.append(planned_file)
    return result


def generation_count_notes(
    non_skipped_count: int,
    generation_row_counts: dict[str, int],
) -> list[str]:
    notes: list[str] = []
    total_rows = sum(generation_row_counts.values())
    if generation_row_counts and total_rows != non_skipped_count:
        notes.append(
            "Generation row count differs from non-skipped attempt count; "
            "this is expected for encdec runs and preserved as metadata."
        )
    return notes


def is_legacy_eval_path(path: Path) -> bool:
    return LEGACY_EVAL_FILE_PATTERN.match(path.name) is not None


def resolve_source_path(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return SOURCE_ROOT / path


def relative_source_path(path: Path) -> str:
    try:
        return str(path.relative_to(SOURCE_ROOT))
    except ValueError:
        return str(path)


def role_for_optimizer_file(path: Path) -> str:
    name = path.name
    if name.endswith("_summary.json"):
        return "optimization_summary"
    if name.endswith("_events.jsonl"):
        return "optimization_events"
    if name.endswith("_run.log"):
        return "optimization_run_log"
    if (
        name.endswith("_optimized.json")
        or "_optimized_" in name
        and name.endswith(".json")
    ):
        return "optimized_program"
    return "optimization_artifact"


def role_for_optimizer_internal_file(path: Path) -> str:
    if path.name == "gepa_state.bin":
        return "optimizer_state"
    if "evaluated_programs" in path.parts:
        return "optimizer_evaluated_program"
    if "generated_best_outputs_valset" in path.parts:
        return "optimizer_generated_best_output"
    return "optimizer_internal_artifact"


def role_for_unknown_file(path: Path) -> str:
    if path.suffix == ".jsonl":
        return "generation_history"
    if path.name == "human_eval_dspy_snapshot_latest.json":
        return "archive_snapshot"
    if path.name == "gepa_state.bin":
        return "optimizer_state"
    if path.suffix == ".log":
        return "run_log"
    return "unassigned_artifact"


def unknown_reason_for_path(path: Path) -> str:
    relative = relative_source_path(path)
    if (
        "dspy_optimized/mipro_logs/" in relative
        or "dspy_optimized/gepa_logs/" in relative
    ):
        return "unlinked_optimizer_internal_dir"
    if "archive/" in relative and path.suffix == ".jsonl":
        return "archive_generation_history_without_explicit_session"
    if path.name == "human_eval_dspy_snapshot_latest.json":
        return "archive_snapshot"
    if path.suffix == ".jsonl":
        return "no_explicit_generation_reference"
    return "unassigned"


def record_count_for_path(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        return count_jsonl_records(path)
    if path.name == "human_eval_dspy_snapshot_latest.json":
        payload = read_json_file(path)
        return len(payload.get("log_files") or [])
    return None


def count_jsonl_records(path: Path) -> int:
    return sum(1 for _ in srsly.read_jsonl(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        value = srsly.read_json(path)
    except ValueError as exc:
        if "Value is too big" not in str(exc):
            raise
        # srsly uses ujson here, which cannot load the 305 MB eval run JSON.
        value = stdlib_json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(value).__name__}")
    return value


def write_metadata(path: Path, metadata: SessionMetadata) -> None:
    write_json_file(path, metadata.model_dump(mode="json"))


def write_json_file(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    srsly.write_json(path, value, indent=2)


if __name__ == "__main__":
    main()
