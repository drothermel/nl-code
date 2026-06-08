from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nl_code.code_execution.models import TestCaseResult


SCHEMA_VERSION = "dspy_eval_session_report_v0"
MANIFEST_SCHEMA_VERSION = "dspy_eval_session_report_manifest_v0"
METADATA_FILE_NAME = "metadata.json"
RAW_DIR_NAME = "raw"
EVAL_RUN_PREFIX = "human_eval_dspy_run_"
LEGACY_EVAL_MARKER = "_eval_"
GENERATION_HISTORY_SUFFIX = ".jsonl"
TASK_ID_SEPARATOR = "/"
DEFAULT_REPORTS_DIR_NAME = "parsed_eval_reports"
REPORT_FILE_SUFFIX = ".eval_report.json"
MANIFEST_FILE_NAME = "manifest.json"


class RunFormat(StrEnum):
    PACKAGE = "package"
    LEGACY_NOTEBOOK = "legacy_notebook"


class GenerationAttemptRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_type: str | None = None
    dataset_index: int | None = None
    task_id: str | None = None
    repeat_index: int | None = None
    call_index: int | None = None


class GenerationCallReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_file: str
    source_relative_path: str
    record_index: int
    timestamp: datetime | None = None
    uuid: str | None = None
    model: str | None = None
    response_model: str | None = None
    model_type: str | None = None
    prompt_fingerprint: str | None = None
    prompt_kind: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    response: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None
    attempt: GenerationAttemptRef | None = None


class AttemptReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    run_format: RunFormat
    source_file: str
    source_relative_path: str
    timestamp: datetime | None = None
    generation_type: str | None = None
    dataset_index: int | None = None
    task_id: str | None = None
    repeat_index: int | None = None
    skipped: bool = False
    error: str | None = None
    code_spec: str | None = None
    raw_completed_code: str = ""
    extracted_code: str = ""
    test_pass_rate: float = 0.0
    test_case_results: list[TestCaseResult] = Field(default_factory=list)
    generation_log_file: str | None = None
    generation_log_source_relative_path: str | None = None
    generation_call_ids: list[str] = Field(default_factory=list)

    @property
    def evaluated(self) -> bool:
        return not self.skipped

    @property
    def passed(self) -> bool:
        return self.evaluated and self.test_pass_rate == 1.0

    @property
    def failed_test_count(self) -> int:
        return sum(not result.passed for result in self.test_case_results)

    @property
    def first_failed_result(self) -> TestCaseResult | None:
        for result in self.test_case_results:
            if not result.passed:
                return result
        return None


class EvalSummaryReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_attempts: int
    evaluated_attempts: int
    skipped_count: int
    attempt_pass_count: int
    attempt_pass_rate: float
    sample_best_pass_count: int
    sample_best_pass_rate: float
    average_test_pass_rate: float


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_file: str
    source_relative_path: str
    run_format: RunFormat
    timestamp: datetime | None = None
    generation_type: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    selected_dataset_indices: list[int] = Field(default_factory=list)
    summaries: dict[str, EvalSummaryReport] = Field(default_factory=dict)
    attempt_ids: list[str] = Field(default_factory=list)
    generation_call_ids: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(default_factory=list)


class SampleReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_order: int
    task_id: str | None = None
    dataset_index: int | None = None
    generation_type: str | None = None
    attempt_ids: list[str]
    attempt_count: int
    evaluated_attempt_count: int
    skipped_count: int
    pass_count: int
    attempt_pass_rate: float
    best_passed: bool
    best_test_pass_rate: float
    average_test_pass_rate: float


class AggregateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_attempts: int
    evaluated_attempts: int
    skipped_count: int
    pass_count: int
    attempt_pass_rate: float
    sample_count: int
    sample_best_pass_count: int
    sample_best_pass_rate: float
    average_test_pass_rate: float


class SessionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    created_at: datetime
    session: dict[str, Any]
    runs: list[RunReport]
    samples: list[SampleReport]
    attempts: list[AttemptReport]
    generation_calls: list[GenerationCallReport]
    aggregates: dict[str, AggregateReport]
    parse_notes: list[str] = Field(default_factory=list)


class CorpusReportFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    session_dir: str
    report_file: str
    run_count: int
    sample_count: int
    attempt_count: int
    generation_call_count: int
    generation_types: list[str]


class CorpusReportManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = MANIFEST_SCHEMA_VERSION
    created_at: datetime
    source_root: str
    output_dir: str
    session_count: int
    eval_session_count: int
    skipped_session_count: int
    reports: list[CorpusReportFile]


app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Session directory, or corpus root when --walk is set.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Write the full forensic report JSON to this path.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help=(
            "Directory for --walk outputs. Defaults to "
            "<corpus-root>/parsed_eval_reports."
        ),
    ),
    walk: bool = typer.Option(
        False,
        "--walk",
        help=(
            "Treat the input directory as a corpus root and write one report per "
            "eval session."
        ),
    ),
    examples: int = typer.Option(
        10,
        "--examples",
        "-n",
        min=0,
        help="Number of samples to render in the terminal.",
    ),
) -> None:
    if walk:
        if output_file is not None:
            raise typer.BadParameter("--output-file cannot be used with --walk")
        manifest = write_corpus_reports(input_dir, output_dir=output_dir)
        render_corpus_manifest(manifest)
        return

    report = build_session_report(input_dir)
    if output_file is not None:
        write_report(report, output_file)
        typer.echo(f"Report: {output_file}")
        return

    render_report(report, examples=examples)


def build_session_report(session_dir: Path) -> SessionReport:
    session_dir = session_dir.resolve()
    metadata_path = session_dir / METADATA_FILE_NAME
    if not metadata_path.exists():
        raise typer.BadParameter(f"session directory must contain {METADATA_FILE_NAME}")

    metadata = read_json_file(metadata_path)
    raw_dir = session_dir / RAW_DIR_NAME
    parse_notes: list[str] = []
    if not raw_dir.exists():
        parse_notes.append(f"missing {RAW_DIR_NAME}/ directory")

    generation_calls = parse_generation_calls(
        raw_dir if raw_dir.exists() else session_dir
    )
    generation_calls_by_source = group_generation_calls_by_source(generation_calls)

    runs: list[RunReport] = []
    attempts: list[AttemptReport] = []
    for path in iter_eval_run_paths(raw_dir if raw_dir.exists() else session_dir):
        run, run_attempts = parse_eval_run(
            path, session_dir, generation_calls_by_source
        )
        runs.append(run)
        attempts.extend(run_attempts)

    attempts_by_id = {attempt.id: attempt for attempt in attempts}
    calls_by_id = {call.id: call for call in generation_calls}
    runs = [
        attach_run_metadata(run, attempts_by_id, calls_by_id)
        for run in sorted(runs, key=lambda item: item.source_relative_path)
    ]
    samples = build_samples(attempts)
    aggregates = build_aggregates(attempts, samples)

    return SessionReport(
        created_at=datetime.now(timezone.utc),
        session=session_metadata(metadata, session_dir),
        runs=runs,
        samples=samples,
        attempts=attempts,
        generation_calls=generation_calls,
        aggregates=aggregates,
        parse_notes=parse_notes,
    )


def write_report(report: SessionReport, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(output_file, report.model_dump(mode="json"))


def write_corpus_reports(
    corpus_dir: Path,
    *,
    output_dir: Path | None,
) -> CorpusReportManifest:
    corpus_dir = corpus_dir.resolve()
    report_dir = (output_dir or corpus_dir / DEFAULT_REPORTS_DIR_NAME).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = iter_session_dirs(corpus_dir)
    report_files = []
    skipped_count = 0
    for session_dir in session_dirs:
        report = build_session_report(session_dir)
        if not report.attempts:
            skipped_count += 1
            continue
        report_path = report_dir / report_file_name(report)
        write_report(report, report_path)
        report_files.append(corpus_report_file(report, session_dir, report_path))

    manifest = CorpusReportManifest(
        created_at=datetime.now(timezone.utc),
        source_root=str(corpus_dir),
        output_dir=str(report_dir),
        session_count=len(session_dirs),
        eval_session_count=len(report_files),
        skipped_session_count=skipped_count,
        reports=report_files,
    )
    write_json_file(report_dir / MANIFEST_FILE_NAME, manifest.model_dump(mode="json"))
    return manifest


def render_report(report: SessionReport, *, examples: int) -> None:
    console = Console()
    session = report.session
    console.print(
        Panel(
            Text(
                "\n".join(
                    [
                        f"Session: {session.get('session_id')}",
                        f"Kind: {session.get('session_kind')}",
                        f"Confidence: {session.get('confidence')}",
                        f"Runs: {len(report.runs)}",
                        f"Samples: {len(report.samples)}",
                        f"Attempts: {len(report.attempts)}",
                        f"Generation calls: {len(report.generation_calls)}",
                    ]
                )
            ),
            title="DSPy Eval Session",
        )
    )

    if report.parse_notes:
        console.print(Panel("\n".join(report.parse_notes), title="Parse Notes"))
    if not report.runs:
        console.print("No evaluation runs found in this session.")
        return

    console.print(aggregate_table(report.aggregates))
    console.print(run_table(report.runs))
    console.print(sample_table(report.samples[:examples], report.attempts))


def render_corpus_manifest(manifest: CorpusReportManifest) -> None:
    console = Console()
    console.print(
        Panel(
            Text(
                "\n".join(
                    [
                        f"Source root: {manifest.source_root}",
                        f"Output dir: {manifest.output_dir}",
                        f"Sessions scanned: {manifest.session_count}",
                        f"Eval sessions written: {manifest.eval_session_count}",
                        f"Skipped sessions: {manifest.skipped_session_count}",
                    ]
                )
            ),
            title="DSPy Eval Corpus Reports",
        )
    )
    table = Table(title="Written Reports")
    table.add_column("Session")
    table.add_column("Runs", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Attempts", justify="right")
    table.add_column("Generation Types")
    table.add_column("Report")
    for report_file in manifest.reports:
        table.add_row(
            report_file.session_id,
            str(report_file.run_count),
            str(report_file.sample_count),
            str(report_file.attempt_count),
            ", ".join(report_file.generation_types),
            Path(report_file.report_file).name,
        )
    console.print(table)


def aggregate_table(aggregates: dict[str, AggregateReport]) -> Table:
    table = Table(title="Aggregates")
    table.add_column("Group")
    table.add_column("Attempts", justify="right")
    table.add_column("Evaluated", justify="right")
    table.add_column("Skipped", justify="right")
    table.add_column("Attempt Pass", justify="right")
    table.add_column("Sample Best", justify="right")
    table.add_column("Avg Test", justify="right")
    for name, aggregate in aggregates.items():
        table.add_row(
            name,
            str(aggregate.total_attempts),
            str(aggregate.evaluated_attempts),
            str(aggregate.skipped_count),
            format_rate(aggregate.attempt_pass_rate),
            format_rate(aggregate.sample_best_pass_rate),
            format_rate(aggregate.average_test_pass_rate),
        )
    return table


def run_table(runs: list[RunReport]) -> Table:
    table = Table(title="Runs")
    table.add_column("Run")
    table.add_column("Format")
    table.add_column("Generation")
    table.add_column("Model")
    table.add_column("Attempts", justify="right")
    table.add_column("Calls", justify="right")
    for run in runs:
        table.add_row(
            run.id,
            run.run_format.value,
            run.generation_type or "",
            ", ".join(run.model_names),
            str(len(run.attempt_ids)),
            str(len(run.generation_call_ids)),
        )
    return table


def sample_table(samples: list[SampleReport], attempts: list[AttemptReport]) -> Table:
    attempts_by_id = {attempt.id: attempt for attempt in attempts}
    table = Table(title="Samples")
    table.add_column("Task")
    table.add_column("Index", justify="right")
    table.add_column("Generation")
    table.add_column("Attempts", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("Avg Test", justify="right")
    table.add_column("Repeats")
    for sample in samples:
        repeat_parts = []
        for attempt_id in sample.attempt_ids:
            attempt = attempts_by_id[attempt_id]
            repeat_label = (
                f"r{attempt.repeat_index}" if attempt.repeat_index is not None else "r?"
            )
            status = "skip" if attempt.skipped else format_rate(attempt.test_pass_rate)
            if attempt.error:
                status = "error"
            repeat_parts.append(
                f"{repeat_label}:{status} "
                f"fails={attempt.failed_test_count} "
                f"calls={len(attempt.generation_call_ids)}"
            )
        table.add_row(
            sample.task_id or "",
            "" if sample.dataset_index is None else str(sample.dataset_index),
            sample.generation_type or "",
            str(sample.attempt_count),
            format_rate(sample.best_test_pass_rate),
            format_rate(sample.average_test_pass_rate),
            "; ".join(repeat_parts),
        )
    return table


def session_metadata(metadata: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    return {
        "session_id": metadata.get("session_id") or session_dir.name,
        "session_kind": metadata.get("session_kind"),
        "confidence": metadata.get("confidence"),
        "session_dir": str(session_dir),
        "created_at": metadata.get("created_at"),
        "source_roots": metadata.get("source_roots") or [],
        "original_grouping_paths": metadata.get("original_grouping_paths") or [],
        "extracted": metadata.get("extracted") or {},
        "files": metadata.get("files") or [],
    }


def report_file_name(report: SessionReport) -> str:
    return f"{report.session['session_id']}{REPORT_FILE_SUFFIX}"


def corpus_report_file(
    report: SessionReport,
    session_dir: Path,
    report_path: Path,
) -> CorpusReportFile:
    generation_types = sorted(
        {
            sample.generation_type
            for sample in report.samples
            if sample.generation_type is not None
        }
    )
    return CorpusReportFile(
        session_id=str(report.session["session_id"]),
        session_dir=str(session_dir),
        report_file=str(report_path),
        run_count=len(report.runs),
        sample_count=len(report.samples),
        attempt_count=len(report.attempts),
        generation_call_count=len(report.generation_calls),
        generation_types=generation_types,
    )


def iter_eval_run_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("human_eval_dspy*.json") if is_eval_run_path(path)
    )


def iter_session_dirs(corpus_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in corpus_dir.iterdir()
        if path.is_dir() and (path / METADATA_FILE_NAME).exists()
    )


def is_eval_run_path(path: Path) -> bool:
    name = path.name
    if name.startswith(EVAL_RUN_PREFIX) and path.suffix == ".json":
        return True
    return (
        path.suffix == ".json"
        and LEGACY_EVAL_MARKER in name
        and (
            name.startswith("human_eval_dspy_direct_eval_")
            or name.startswith("human_eval_dspy_encdec_eval_")
        )
    )


def parse_eval_run(
    path: Path,
    session_dir: Path,
    generation_calls_by_source: dict[str, list[GenerationCallReport]],
) -> tuple[RunReport, list[AttemptReport]]:
    payload = read_json_file(path)
    if "attempts" in payload:
        return parse_package_eval_run(
            path, session_dir, payload, generation_calls_by_source
        )
    return parse_legacy_eval_run(path, session_dir, payload, generation_calls_by_source)


def parse_package_eval_run(
    path: Path,
    session_dir: Path,
    payload: dict[str, Any],
    generation_calls_by_source: dict[str, list[GenerationCallReport]],
) -> tuple[RunReport, list[AttemptReport]]:
    run_id = path.stem
    timestamp = payload.get("timestamp")
    config = payload.get("config") or {}
    attempts = [
        package_attempt(
            path,
            session_dir,
            run_id,
            timestamp,
            index,
            raw_attempt,
            generation_calls_by_source,
        )
        for index, raw_attempt in enumerate(payload.get("attempts") or [])
    ]
    summaries = {
        str(name): EvalSummaryReport.model_validate(summary)
        for name, summary in (payload.get("summaries") or {}).items()
    }
    if not summaries:
        summaries = summaries_for_attempts(attempts)
    run = RunReport(
        id=run_id,
        source_file=str(path),
        source_relative_path=relative_to_session(path, session_dir),
        run_format=RunFormat.PACKAGE,
        timestamp=timestamp,
        generation_type=config.get("generation_type"),
        config=config,
        selected_dataset_indices=payload.get("selected_dataset_indices") or [],
        summaries=summaries,
        attempt_ids=[attempt.id for attempt in attempts],
    )
    return run, attempts


def parse_legacy_eval_run(
    path: Path,
    session_dir: Path,
    payload: dict[str, Any],
    generation_calls_by_source: dict[str, list[GenerationCallReport]],
) -> tuple[RunReport, list[AttemptReport]]:
    run_id = path.stem
    timestamp = payload.get("timestamp")
    generation_type = payload.get("eval_type")
    attempts = [
        legacy_attempt(
            path,
            session_dir,
            run_id,
            timestamp,
            generation_type,
            index,
            raw_item,
            generation_calls_by_source,
        )
        for index, raw_item in enumerate(payload.get("outputs") or [])
    ]
    summaries = summaries_for_attempts(attempts)
    run = RunReport(
        id=run_id,
        source_file=str(path),
        source_relative_path=relative_to_session(path, session_dir),
        run_format=RunFormat.LEGACY_NOTEBOOK,
        timestamp=timestamp,
        generation_type=generation_type,
        config={
            "generation_type": generation_type,
            "n_samples": payload.get("val_num"),
            "seed": payload.get("seed"),
        },
        selected_dataset_indices=payload.get("dataset_indices") or [],
        summaries=summaries,
        attempt_ids=[attempt.id for attempt in attempts],
    )
    return run, attempts


def package_attempt(
    path: Path,
    session_dir: Path,
    run_id: str,
    timestamp: Any,
    index: int,
    raw_attempt: dict[str, Any],
    generation_calls_by_source: dict[str, list[GenerationCallReport]],
) -> AttemptReport:
    generation_log_file = raw_attempt.get("generation_log_file")
    generation_log_source = source_relative_path_for_ref(
        generation_log_file,
        generation_calls_by_source,
    )
    attempt = AttemptReport(
        id=attempt_id(run_id, index),
        run_id=run_id,
        run_format=RunFormat.PACKAGE,
        source_file=str(path),
        source_relative_path=relative_to_session(path, session_dir),
        timestamp=timestamp,
        generation_type=raw_attempt.get("generation_type"),
        dataset_index=raw_attempt.get("dataset_index"),
        task_id=raw_attempt.get("task_id"),
        repeat_index=raw_attempt.get("repeat_index"),
        skipped=bool(raw_attempt.get("skipped", False)),
        error=raw_attempt.get("error"),
        code_spec=raw_attempt.get("code_spec"),
        raw_completed_code=raw_attempt.get("raw_completed_code") or "",
        extracted_code=raw_attempt.get("extracted_code") or "",
        test_pass_rate=float(raw_attempt.get("test_pass_rate") or 0.0),
        test_case_results=[
            TestCaseResult.model_validate(result)
            for result in raw_attempt.get("test_case_results", [])
        ],
        generation_log_file=generation_log_file,
        generation_log_source_relative_path=generation_log_source,
    )
    return attempt.model_copy(
        update={
            "generation_call_ids": matched_generation_call_ids(
                attempt,
                generation_log_source,
                generation_calls_by_source,
            )
        }
    )


def legacy_attempt(
    path: Path,
    session_dir: Path,
    run_id: str,
    timestamp: Any,
    generation_type: str | None,
    index: int,
    raw_item: dict[str, Any],
    generation_calls_by_source: dict[str, list[GenerationCallReport]],
) -> AttemptReport:
    output = raw_item.get("output") or {}
    prediction = output.get("prediction") or {}
    generation_log_file = output.get("log_file")
    generation_log_source = source_relative_path_for_ref(
        generation_log_file,
        generation_calls_by_source,
    )
    attempt = AttemptReport(
        id=attempt_id(run_id, index),
        run_id=run_id,
        run_format=RunFormat.LEGACY_NOTEBOOK,
        source_file=str(path),
        source_relative_path=relative_to_session(path, session_dir),
        timestamp=timestamp,
        generation_type=generation_type,
        dataset_index=raw_item.get("dataset_index"),
        task_id=output.get("task_id"),
        repeat_index=0,
        skipped=bool(output.get("skipped", False)),
        error=output.get("error"),
        code_spec=prediction.get("code_spec"),
        raw_completed_code=prediction.get("completed_code") or "",
        extracted_code=output.get("extracted") or "",
        test_pass_rate=float(output.get("pass_rate") or 0.0),
        test_case_results=[
            TestCaseResult.model_validate(result)
            for result in output.get("results", [])
        ],
        generation_log_file=generation_log_file,
        generation_log_source_relative_path=generation_log_source,
    )
    return attempt.model_copy(
        update={
            "generation_call_ids": matched_generation_call_ids(
                attempt,
                generation_log_source,
                generation_calls_by_source,
            )
        }
    )


def parse_generation_calls(root: Path) -> list[GenerationCallReport]:
    calls = []
    for path in sorted(root.rglob(f"human_eval_dspy*{GENERATION_HISTORY_SUFFIX}")):
        source_relative_path = source_relative_path_for_session_file(path)
        for record_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            record = parse_json_text(line)
            calls.append(
                generation_call_report(
                    path=path,
                    source_relative_path=source_relative_path,
                    record_index=record_index,
                    record=record,
                )
            )
    return calls


def generation_call_report(
    *,
    path: Path,
    source_relative_path: str,
    record_index: int,
    record: dict[str, Any],
) -> GenerationCallReport:
    messages = record.get("messages") or []
    response = record.get("response") or {}
    usage = record.get("usage") or {}
    return GenerationCallReport(
        id=generation_call_id(source_relative_path, record_index),
        source_file=str(path),
        source_relative_path=source_relative_path,
        record_index=record_index,
        timestamp=record.get("timestamp"),
        uuid=record.get("uuid"),
        model=record.get("model"),
        response_model=record.get("response_model") or response.get("model"),
        model_type=record.get("model_type"),
        prompt_fingerprint=fingerprint_messages(messages),
        prompt_kind=prompt_kind(messages),
        messages=messages,
        outputs=record.get("outputs") or response_outputs(response),
        response=response,
        usage=usage,
        cost=record.get("cost") or usage.get("cost"),
        attempt=record.get("attempt"),
    )


def group_generation_calls_by_source(
    calls: list[GenerationCallReport],
) -> dict[str, list[GenerationCallReport]]:
    grouped: dict[str, list[GenerationCallReport]] = defaultdict(list)
    for call in calls:
        grouped[call.source_relative_path].append(call)
    return dict(grouped)


def matched_generation_call_ids(
    attempt: AttemptReport,
    generation_log_source: str | None,
    generation_calls_by_source: dict[str, list[GenerationCallReport]],
) -> list[str]:
    if generation_log_source is None:
        candidate_calls = [
            call for calls in generation_calls_by_source.values() for call in calls
        ]
    else:
        candidate_calls = generation_calls_by_source.get(generation_log_source, [])
    return [
        call.id
        for call in candidate_calls
        if generation_call_matches_attempt(call, attempt)
    ]


def generation_call_matches_attempt(
    call: GenerationCallReport,
    attempt: AttemptReport,
) -> bool:
    if call.attempt is None:
        return False
    if call.attempt.task_id != attempt.task_id:
        return False
    if call.attempt.generation_type != attempt.generation_type:
        return False
    if call.attempt.repeat_index != attempt.repeat_index:
        return False
    return (
        call.attempt.dataset_index is None
        or attempt.dataset_index is None
        or call.attempt.dataset_index == attempt.dataset_index
    )


def attach_run_metadata(
    run: RunReport,
    attempts_by_id: dict[str, AttemptReport],
    calls_by_id: dict[str, GenerationCallReport],
) -> RunReport:
    run_attempts = [attempts_by_id[attempt_id] for attempt_id in run.attempt_ids]
    call_ids = sorted(
        {call_id for attempt in run_attempts for call_id in attempt.generation_call_ids}
    )
    model_names = sorted(
        {
            calls_by_id[call_id].model
            for call_id in call_ids
            if calls_by_id[call_id].model is not None
        }
    )
    return run.model_copy(
        update={
            "generation_call_ids": call_ids,
            "model_names": model_names,
        }
    )


def build_samples(attempts: list[AttemptReport]) -> list[SampleReport]:
    grouped: dict[tuple[str | None, int | None, str | None], list[AttemptReport]]
    grouped = defaultdict(list)
    first_order: dict[tuple[str | None, int | None, str | None], int] = {}
    for order, attempt in enumerate(attempts):
        key = (attempt.task_id, attempt.dataset_index, attempt.generation_type)
        grouped[key].append(attempt)
        first_order.setdefault(key, order)

    samples = [
        sample_report(key, key_attempts, first_order[key])
        for key, key_attempts in grouped.items()
    ]
    return sorted(samples, key=lambda sample: sample.source_order)


def sample_report(
    key: tuple[str | None, int | None, str | None],
    attempts: list[AttemptReport],
    source_order: int,
) -> SampleReport:
    task_id, dataset_index, generation_type = key
    evaluated = [attempt for attempt in attempts if attempt.evaluated]
    pass_count = sum(attempt.passed for attempt in evaluated)
    best_pass_rate = max(
        (attempt.test_pass_rate for attempt in evaluated),
        default=0.0,
    )
    average_pass_rate = (
        sum(attempt.test_pass_rate for attempt in evaluated) / len(evaluated)
        if evaluated
        else 0.0
    )
    return SampleReport(
        id=sample_id(task_id, dataset_index, generation_type),
        source_order=source_order,
        task_id=task_id,
        dataset_index=dataset_index,
        generation_type=generation_type,
        attempt_ids=[attempt.id for attempt in attempts],
        attempt_count=len(attempts),
        evaluated_attempt_count=len(evaluated),
        skipped_count=len(attempts) - len(evaluated),
        pass_count=pass_count,
        attempt_pass_rate=pass_count / len(evaluated) if evaluated else 0.0,
        best_passed=any(attempt.passed for attempt in evaluated),
        best_test_pass_rate=best_pass_rate,
        average_test_pass_rate=average_pass_rate,
    )


def build_aggregates(
    attempts: list[AttemptReport],
    samples: list[SampleReport],
) -> dict[str, AggregateReport]:
    aggregates = {"overall": aggregate_report(attempts, samples)}
    generation_types = sorted(
        {
            attempt.generation_type
            for attempt in attempts
            if attempt.generation_type is not None
        }
    )
    for generation_type in generation_types:
        generation_attempts = [
            attempt
            for attempt in attempts
            if attempt.generation_type == generation_type
        ]
        generation_samples = [
            sample for sample in samples if sample.generation_type == generation_type
        ]
        aggregates[generation_type] = aggregate_report(
            generation_attempts,
            generation_samples,
        )
    return aggregates


def aggregate_report(
    attempts: list[AttemptReport],
    samples: list[SampleReport],
) -> AggregateReport:
    evaluated = [attempt for attempt in attempts if attempt.evaluated]
    pass_count = sum(attempt.passed for attempt in evaluated)
    sample_best_pass_count = sum(sample.best_passed for sample in samples)
    return AggregateReport(
        total_attempts=len(attempts),
        evaluated_attempts=len(evaluated),
        skipped_count=len(attempts) - len(evaluated),
        pass_count=pass_count,
        attempt_pass_rate=pass_count / len(evaluated) if evaluated else 0.0,
        sample_count=len(samples),
        sample_best_pass_count=sample_best_pass_count,
        sample_best_pass_rate=(
            sample_best_pass_count / len(samples) if samples else 0.0
        ),
        average_test_pass_rate=(
            sum(attempt.test_pass_rate for attempt in evaluated) / len(evaluated)
            if evaluated
            else 0.0
        ),
    )


def summaries_for_attempts(
    attempts: list[AttemptReport],
) -> dict[str, EvalSummaryReport]:
    generation_types = sorted(
        {
            attempt.generation_type
            for attempt in attempts
            if attempt.generation_type is not None
        }
    )
    return {
        generation_type: summary_for_attempts(
            [
                attempt
                for attempt in attempts
                if attempt.generation_type == generation_type
            ]
        )
        for generation_type in generation_types
    }


def summary_for_attempts(attempts: list[AttemptReport]) -> EvalSummaryReport:
    evaluated = [attempt for attempt in attempts if attempt.evaluated]
    pass_count = sum(attempt.passed for attempt in evaluated)
    sample_passed: dict[str, bool] = {}
    for attempt in evaluated:
        sample_passed[sample_id(attempt.task_id, attempt.dataset_index, None)] = (
            sample_passed.get(
                sample_id(attempt.task_id, attempt.dataset_index, None), False
            )
            or attempt.passed
        )
    return EvalSummaryReport(
        total_attempts=len(attempts),
        evaluated_attempts=len(evaluated),
        skipped_count=len(attempts) - len(evaluated),
        attempt_pass_count=pass_count,
        attempt_pass_rate=pass_count / len(evaluated) if evaluated else 0.0,
        sample_best_pass_count=sum(sample_passed.values()),
        sample_best_pass_rate=(
            sum(sample_passed.values()) / len(sample_passed) if sample_passed else 0.0
        ),
        average_test_pass_rate=(
            sum(attempt.test_pass_rate for attempt in evaluated) / len(evaluated)
            if evaluated
            else 0.0
        ),
    )


def source_relative_path_for_ref(
    value: str | None,
    generation_calls_by_source: dict[str, list[GenerationCallReport]],
) -> str | None:
    if value is None:
        return None
    normalized_value = value.removeprefix(f"{RAW_DIR_NAME}/")
    if normalized_value in generation_calls_by_source:
        return normalized_value
    path = Path(value)
    path_parts = list(path.parts)
    for index, part in enumerate(path_parts):
        candidate = "/".join(path_parts[index:])
        if candidate in generation_calls_by_source:
            return candidate
    name_matches = [
        source
        for source in generation_calls_by_source
        if Path(source).name == path.name
    ]
    return name_matches[0] if len(name_matches) == 1 else None


def read_json_file(path: Path) -> dict[str, Any]:
    return parse_json_text(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def parse_json_text(value: str) -> dict[str, Any]:
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


def source_relative_path_for_session_file(path: Path) -> str:
    parts = list(path.parts)
    if RAW_DIR_NAME in parts:
        raw_index = parts.index(RAW_DIR_NAME)
        return "/".join(parts[raw_index + 1 :])
    return path.name


def relative_to_session(path: Path, session_dir: Path) -> str:
    try:
        return str(path.relative_to(session_dir))
    except ValueError:
        return str(path)


def attempt_id(run_id: str, index: int) -> str:
    return f"{run_id}:attempt:{index:06d}"


def generation_call_id(source_relative_path: str, record_index: int) -> str:
    return f"{source_relative_path}:record:{record_index:06d}"


def sample_id(
    task_id: str | None,
    dataset_index: int | None,
    generation_type: str | None,
) -> str:
    task_part = (task_id or f"dataset_index={dataset_index}").replace(
        TASK_ID_SEPARATOR,
        "_",
    )
    generation_part = generation_type or "unknown_generation"
    return f"{task_part}:{generation_part}"


def fingerprint_messages(messages: list[dict[str, Any]]) -> str | None:
    prompt_parts = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    ]
    if not prompt_parts:
        prompt_parts = [
            str(message.get("role", "")) for message in messages if message.get("role")
        ]
    if not prompt_parts:
        return None
    return hashlib.sha256("\n\n".join(prompt_parts).encode("utf-8")).hexdigest()


def prompt_kind(messages: list[dict[str, Any]]) -> str | None:
    system_content = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    )
    if "`code_stub`" in system_content:
        return "direct_code_from_stub"
    if "`input_code`" in system_content and "`code_spec`" in system_content:
        return "encode_code_spec"
    if "`code_spec`" in system_content and "`function_stub`" in system_content:
        return "decode_code_spec"
    return None


def response_outputs(response: dict[str, Any]) -> list[str]:
    outputs = []
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            outputs.append(content)
    return outputs


def format_rate(value: float) -> str:
    return f"{value:.1%}"


if __name__ == "__main__":
    app()
