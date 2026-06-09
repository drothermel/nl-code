from __future__ import annotations

import ast
import hashlib
import json
import pickletools
import re
from collections import Counter
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


SCHEMA_VERSION = "dspy_gepa_session_report_v0"
MANIFEST_SCHEMA_VERSION = "dspy_gepa_session_report_manifest_v0"
METADATA_FILE_NAME = "metadata.json"
RAW_DIR_NAME = "raw"
DEFAULT_REPORTS_DIR_NAME = "parsed_gepa_reports"
REPORT_FILE_SUFFIX = ".gepa_report.json"
MANIFEST_FILE_NAME = "manifest.json"
SUMMARY_SUFFIX = "_summary.json"
GENERATED_BEST_OUTPUTS_DIR = "generated_best_outputs_valset"
GEPA_STATE_FILE_NAME = "gepa_state.bin"
GEPA_LOG_DIR_MARKER = "gepa_logs"
UNSAFE_PICKLE_OPS = {
    "BUILD",
    "EXT1",
    "EXT2",
    "EXT4",
    "GLOBAL",
    "INST",
    "NEWOBJ",
    "NEWOBJ_EX",
    "OBJ",
    "REDUCE",
    "STACK_GLOBAL",
}
RUN_LOG_TIMESTAMP_RE = re.compile(
    r"^(?:\r?GEPA Optimization:.*)?\d{4}/\d{2}/\d{2} .* INFO "
    r"dspy\.teleprompt\.gepa\.gepa:"
)
PROPOSAL_RE = re.compile(
    r"Iteration (?P<iteration>\d+): Proposed new text for "
    r"(?P<predictor>[^:]+): (?P<text>.*)"
)
FLOAT_RE = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"


class ExtractionConfidence(StrEnum):
    DEFINITELY_EXTRACTABLE = "definitely_extractable"
    EXTRACTABLE_WITH_INFERENCE = "extractable_with_inference"
    BEST_EFFORT = "best_effort"
    NOT_PRESENT = "not_present"


class ProgramPhase(StrEnum):
    BASELINE = "baseline"
    OPTIMIZED = "optimized"
    CANDIDATE = "candidate"


class MetricCallKind(StrEnum):
    EVAL = "metric_call"
    GEPA = "gepa_metric_call"


class GepaSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file: str
    source_relative_path: str
    line_number: int | None = None
    json_path: str | None = None


class GepaOptimizerRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_file: str
    source_relative_path: str
    timestamp: datetime | None = None
    generation_type: str | None = None
    optimization_target: str | None = None
    model: str | None = None
    llm_config_id: str | None = None
    reasoning_config: dict[str, Any] | None = None
    auto: str | None = None
    max_metric_calls: int | None = None
    num_threads: int | None = None
    seed: int | None = None
    split_task_ids: dict[str, list[str]] = Field(default_factory=dict)
    artifact_paths: dict[str, str | None] = Field(default_factory=dict)
    final_program_id: str | None = None


class GepaProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    phase: ProgramPhase
    candidate_index: int | None = None
    predictor_name: str | None = None
    parent_program_id: str | None = None
    source: GepaSourceRef
    signature_instructions: str | None = None
    signature_fields: list[dict[str, Any]] = Field(default_factory=list)
    demos: list[Any] = Field(default_factory=list)
    train: list[Any] = Field(default_factory=list)
    traces: list[Any] = Field(default_factory=list)
    lm: Any = None
    dependency_versions: dict[str, Any] = Field(default_factory=dict)
    confidence: ExtractionConfidence


class GepaSplitEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    phase: ProgramPhase
    split: str
    label: str
    program_id: str | None = None
    task_count: int
    average_pass_rate: float
    full_pass_count: int
    full_pass_rate: float
    source: GepaSourceRef
    confidence: ExtractionConfidence


class GepaTaskScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    evaluation_id: str
    phase: ProgramPhase
    split: str
    task_id: str
    program_id: str | None = None
    valset_index: int | None = None
    pass_rate: float
    error: str | None = None
    source: GepaSourceRef
    confidence: ExtractionConfidence


class GepaMetricCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    kind: MetricCallKind
    timestamp: datetime | None = None
    label: str | None = None
    metric_call: int | None = None
    phase: str | None = None
    split: str | None = None
    iteration: int | None = None
    program_id: str | None = None
    predictor: str | None = None
    task_id: str | None = None
    pass_rate: float | None = None
    error: str | None = None
    source: GepaSourceRef
    confidence: ExtractionConfidence


class GepaGeneratedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    program_id: str | None = None
    task_id: str | None = None
    valset_index: int | None = None
    iteration: int | None = None
    program_index: int | None = None
    output_fields: dict[str, Any] = Field(default_factory=dict)
    completed_code: str | None = None
    source: GepaSourceRef
    confidence: ExtractionConfidence


class GepaOptimizerIteration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    iteration: int
    selected_program_index: int | None = None
    selected_score: float | None = None
    proposed_predictor: str | None = None
    proposal_text: str | None = None
    proposal_char_count: int | None = None
    new_subsample_score: float | None = None
    old_subsample_score: float | None = None
    subsample_comparison: str | None = None
    subsample_action: str | None = None
    valset_score: float | None = None
    valset_coverage: str | None = None
    individual_valset_scores: dict[int, float] = Field(default_factory=dict)
    pareto_front_scores: dict[int, float] = Field(default_factory=dict)
    pareto_front_programs: dict[int, list[int]] = Field(default_factory=dict)
    pareto_aggregate_score: float | None = None
    best_program_index: int | None = None
    best_score: float | None = None
    linear_pareto_program_index: int | None = None
    new_candidate_index: int | None = None
    found_better_score: float | None = None
    skip_messages: list[str] = Field(default_factory=list)
    source_lines: list[int] = Field(default_factory=list)
    confidence: ExtractionConfidence = ExtractionConfidence.BEST_EFFORT


class GepaCandidateRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    iteration: int
    candidate_index: int | None = None
    valset_score: float | None = None
    best_program_index: int | None = None
    best_score: float | None = None
    pareto_aggregate_score: float | None = None
    linear_pareto_program_index: int | None = None
    source_lines: list[int] = Field(default_factory=list)
    confidence: ExtractionConfidence = ExtractionConfidence.BEST_EFFORT


class GepaStateFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file: str
    source_relative_path: str
    size_bytes: int
    sha256: str | None = None
    decoded: bool = False
    pickle_protocol: int | None = None
    opcode_counts: dict[str, int] = Field(default_factory=dict)
    unsafe_opcode_names: list[str] = Field(default_factory=list)
    string_keys_seen: list[str] = Field(default_factory=list)
    parse_error: str | None = None


class GepaBestEffort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer_iterations: list[GepaOptimizerIteration] = Field(default_factory=list)
    candidate_rankings: list[GepaCandidateRanking] = Field(default_factory=list)
    candidate_valset_scores: list[dict[str, Any]] = Field(default_factory=list)
    candidate_proposals: list[dict[str, Any]] = Field(default_factory=list)
    state_key_inventory: list[dict[str, Any]] = Field(default_factory=list)


class GepaAggregateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_count: int
    program_count: int
    split_evaluation_count: int
    task_score_count: int
    metric_call_count: int
    generated_output_count: int
    optimizer_iteration_count: int
    state_file_count: int


class GepaSessionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    created_at: datetime
    session: dict[str, Any]
    optimizer_runs: list[GepaOptimizerRun]
    programs: list[GepaProgram]
    split_evaluations: list[GepaSplitEvaluation]
    task_scores: list[GepaTaskScore]
    metric_calls: list[GepaMetricCall]
    generated_outputs: list[GepaGeneratedOutput]
    optimizer_iterations: list[GepaOptimizerIteration]
    best_effort: GepaBestEffort
    state_files: list[GepaStateFile]
    aggregates: GepaAggregateReport
    parse_notes: list[str] = Field(default_factory=list)


class GepaCorpusReportFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    session_dir: str
    report_file: str
    optimizer_run_count: int
    program_count: int
    metric_call_count: int
    generated_output_count: int
    generation_types: list[str]


class GepaCorpusReportManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = MANIFEST_SCHEMA_VERSION
    created_at: datetime
    source_root: str
    output_dir: str
    session_count: int
    gepa_session_count: int
    skipped_session_count: int
    reports: list[GepaCorpusReportFile]


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
        help="Write the full GEPA forensic report JSON to this path.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help=(
            "Directory for --walk outputs. Defaults to "
            "<corpus-root>/parsed_gepa_reports."
        ),
    ),
    walk: bool = typer.Option(
        False,
        "--walk",
        help=(
            "Treat the input directory as a corpus root and write one report per "
            "GEPA optimization session."
        ),
    ),
    examples: int = typer.Option(
        10,
        "--examples",
        "-n",
        min=0,
        help="Number of records to render in the terminal.",
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


def build_session_report(session_dir: Path) -> GepaSessionReport:
    session_dir = session_dir.resolve()
    metadata_path = session_dir / METADATA_FILE_NAME
    if not metadata_path.exists():
        raise typer.BadParameter(f"session directory must contain {METADATA_FILE_NAME}")

    metadata = read_json_file(metadata_path)
    raw_dir = session_dir / RAW_DIR_NAME
    parse_notes: list[str] = []
    if not raw_dir.exists():
        parse_notes.append(f"missing {RAW_DIR_NAME}/ directory")

    root = raw_dir if raw_dir.exists() else session_dir
    summary_paths = iter_gepa_summary_paths(root)
    optimizer_runs: list[GepaOptimizerRun] = []
    programs: list[GepaProgram] = []
    split_evaluations: list[GepaSplitEvaluation] = []
    task_scores: list[GepaTaskScore] = []
    metric_calls: list[GepaMetricCall] = []
    generated_outputs: list[GepaGeneratedOutput] = []
    optimizer_iterations: list[GepaOptimizerIteration] = []
    candidate_rankings: list[GepaCandidateRanking] = []
    candidate_valset_scores: list[dict[str, Any]] = []
    candidate_proposals: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        summary = read_json_file(summary_path)
        run_id = run_id_for_summary(summary_path)
        final_program_id = f"{run_id}:program:optimized"
        optimizer_runs.append(
            optimizer_run_report(
                session_dir=session_dir,
                path=summary_path,
                payload=summary,
                run_id=run_id,
                final_program_id=final_program_id,
            )
        )
        programs.extend(
            program_reports(
                session_dir=session_dir,
                summary_path=summary_path,
                summary=summary,
                run_id=run_id,
                final_program_id=final_program_id,
                parse_notes=parse_notes,
            )
        )
        split_reports, score_reports = split_score_reports(
            session_dir=session_dir,
            source_path=summary_path,
            summary=summary,
            run_id=run_id,
            final_program_id=final_program_id,
        )
        split_evaluations.extend(split_reports)
        task_scores.extend(score_reports)
        metric_calls.extend(
            metric_call_reports(
                session_dir=session_dir,
                summary_path=summary_path,
                summary=summary,
                run_id=run_id,
                final_program_id=final_program_id,
                parse_notes=parse_notes,
            )
        )
        run_iterations = optimizer_iteration_reports(
            session_dir=session_dir,
            summary_path=summary_path,
            summary=summary,
            run_id=run_id,
            parse_notes=parse_notes,
        )
        optimizer_iterations.extend(run_iterations)
        programs.extend(
            candidate_program_reports(
                session_dir=session_dir,
                summary_path=summary_path,
                summary=summary,
                iterations=run_iterations,
            )
        )
        candidate_rankings.extend(candidate_rankings_for_iterations(run_iterations))
        candidate_valset_scores.extend(
            candidate_valset_scores_for_iterations(run_iterations)
        )
        candidate_proposals.extend(candidate_proposals_for_iterations(run_iterations))
        generated_outputs.extend(
            generated_output_reports(
                session_dir=session_dir,
                summary_path=summary_path,
                summary=summary,
                run_id=run_id,
                final_program_id=final_program_id,
                parse_notes=parse_notes,
            )
        )

    state_files = [
        inspect_state_file(path, session_dir)
        for path in sorted(root.rglob(GEPA_STATE_FILE_NAME))
    ]
    state_key_inventory = [
        {
            "source_file": state_file.source_file,
            "source_relative_path": state_file.source_relative_path,
            "string_keys_seen": state_file.string_keys_seen,
            "unsafe_opcode_names": state_file.unsafe_opcode_names,
        }
        for state_file in state_files
    ]
    parse_notes.extend(absence_notes())

    aggregate = GepaAggregateReport(
        run_count=len(optimizer_runs),
        program_count=len(programs),
        split_evaluation_count=len(split_evaluations),
        task_score_count=len(task_scores),
        metric_call_count=len(metric_calls),
        generated_output_count=len(generated_outputs),
        optimizer_iteration_count=len(optimizer_iterations),
        state_file_count=len(state_files),
    )
    return GepaSessionReport(
        created_at=datetime.now(timezone.utc),
        session=session_metadata(metadata, session_dir),
        optimizer_runs=optimizer_runs,
        programs=programs,
        split_evaluations=split_evaluations,
        task_scores=task_scores,
        metric_calls=metric_calls,
        generated_outputs=generated_outputs,
        optimizer_iterations=optimizer_iterations,
        best_effort=GepaBestEffort(
            optimizer_iterations=optimizer_iterations,
            candidate_rankings=candidate_rankings,
            candidate_valset_scores=candidate_valset_scores,
            candidate_proposals=candidate_proposals,
            state_key_inventory=state_key_inventory,
        ),
        state_files=state_files,
        aggregates=aggregate,
        parse_notes=parse_notes,
    )


def write_report(report: GepaSessionReport, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(output_file, report.model_dump(mode="json"))


def write_corpus_reports(
    corpus_dir: Path,
    *,
    output_dir: Path | None,
) -> GepaCorpusReportManifest:
    corpus_dir = corpus_dir.resolve()
    report_dir = (output_dir or corpus_dir / DEFAULT_REPORTS_DIR_NAME).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = iter_session_dirs(corpus_dir)
    report_files = []
    skipped_count = 0
    for session_dir in session_dirs:
        report = build_session_report(session_dir)
        if not report.optimizer_runs:
            skipped_count += 1
            continue
        report_path = report_dir / report_file_name(report)
        write_report(report, report_path)
        report_files.append(corpus_report_file(report, session_dir, report_path))

    manifest = GepaCorpusReportManifest(
        created_at=datetime.now(timezone.utc),
        source_root=str(corpus_dir),
        output_dir=str(report_dir),
        session_count=len(session_dirs),
        gepa_session_count=len(report_files),
        skipped_session_count=skipped_count,
        reports=report_files,
    )
    write_json_file(report_dir / MANIFEST_FILE_NAME, manifest.model_dump(mode="json"))
    return manifest


def render_report(report: GepaSessionReport, *, examples: int) -> None:
    console = Console()
    session = report.session
    console.print(
        Panel(
            Text(
                "\n".join(
                    [
                        f"Session: {session.get('session_id')}",
                        f"Kind: {session.get('session_kind')}",
                        f"Runs: {len(report.optimizer_runs)}",
                        f"Programs: {len(report.programs)}",
                        f"Metric calls: {len(report.metric_calls)}",
                        f"Generated outputs: {len(report.generated_outputs)}",
                        f"Best-effort iterations: {len(report.optimizer_iterations)}",
                    ]
                )
            ),
            title="DSPy GEPA Session",
        )
    )
    if report.parse_notes:
        console.print(Panel("\n".join(report.parse_notes), title="Parse Notes"))
    if not report.optimizer_runs:
        console.print("No GEPA optimization summaries found in this session.")
        return
    console.print(run_table(report.optimizer_runs))
    console.print(split_table(report.split_evaluations[:examples]))
    console.print(iteration_table(report.optimizer_iterations[:examples]))


def render_corpus_manifest(manifest: GepaCorpusReportManifest) -> None:
    console = Console()
    console.print(
        Panel(
            Text(
                "\n".join(
                    [
                        f"Source root: {manifest.source_root}",
                        f"Output dir: {manifest.output_dir}",
                        f"Sessions scanned: {manifest.session_count}",
                        f"GEPA sessions written: {manifest.gepa_session_count}",
                        f"Skipped sessions: {manifest.skipped_session_count}",
                    ]
                )
            ),
            title="DSPy GEPA Corpus Reports",
        )
    )
    table = Table(title="Written Reports")
    table.add_column("Session")
    table.add_column("Runs", justify="right")
    table.add_column("Programs", justify="right")
    table.add_column("Metric Calls", justify="right")
    table.add_column("Generation Types")
    table.add_column("Report")
    for report_file in manifest.reports:
        table.add_row(
            report_file.session_id,
            str(report_file.optimizer_run_count),
            str(report_file.program_count),
            str(report_file.metric_call_count),
            ", ".join(report_file.generation_types),
            Path(report_file.report_file).name,
        )
    console.print(table)


def run_table(runs: list[GepaOptimizerRun]) -> Table:
    table = Table(title="GEPA Runs")
    table.add_column("Run")
    table.add_column("Generation")
    table.add_column("Target")
    table.add_column("Model")
    table.add_column("Budget", justify="right")
    table.add_column("Seed", justify="right")
    for run in runs:
        table.add_row(
            run.id,
            run.generation_type or "",
            run.optimization_target or "",
            run.model or "",
            "" if run.max_metric_calls is None else str(run.max_metric_calls),
            "" if run.seed is None else str(run.seed),
        )
    return table


def split_table(splits: list[GepaSplitEvaluation]) -> Table:
    table = Table(title="Split Evaluations")
    table.add_column("Phase")
    table.add_column("Split")
    table.add_column("Tasks", justify="right")
    table.add_column("Average", justify="right")
    table.add_column("Full Pass", justify="right")
    for split in splits:
        table.add_row(
            split.phase.value,
            split.split,
            str(split.task_count),
            format_rate(split.average_pass_rate),
            format_rate(split.full_pass_rate),
        )
    return table


def iteration_table(iterations: list[GepaOptimizerIteration]) -> Table:
    table = Table(title="Best-Effort GEPA Iterations")
    table.add_column("Iteration", justify="right")
    table.add_column("Selected", justify="right")
    table.add_column("New Candidate", justify="right")
    table.add_column("Valset", justify="right")
    table.add_column("Best", justify="right")
    table.add_column("Proposal", justify="right")
    for iteration in iterations:
        table.add_row(
            str(iteration.iteration),
            ""
            if iteration.selected_program_index is None
            else str(iteration.selected_program_index),
            ""
            if iteration.new_candidate_index is None
            else str(iteration.new_candidate_index),
            ""
            if iteration.valset_score is None
            else format_rate(iteration.valset_score),
            "" if iteration.best_score is None else format_rate(iteration.best_score),
            ""
            if iteration.proposal_char_count is None
            else str(iteration.proposal_char_count),
        )
    return table


def optimizer_run_report(
    *,
    session_dir: Path,
    path: Path,
    payload: dict[str, Any],
    run_id: str,
    final_program_id: str,
) -> GepaOptimizerRun:
    return GepaOptimizerRun(
        id=run_id,
        source_file=str(path),
        source_relative_path=relative_to_session(path, session_dir),
        timestamp=payload.get("timestamp"),
        generation_type=payload.get("generation_type"),
        optimization_target=payload.get("optimization_target"),
        model=payload.get("model"),
        llm_config_id=payload.get("llm_config_id"),
        reasoning_config=payload.get("reasoning_config"),
        auto=payload.get("auto"),
        max_metric_calls=payload.get("max_metric_calls"),
        num_threads=payload.get("num_threads"),
        seed=payload.get("seed"),
        split_task_ids={
            "train": payload.get("train_task_ids") or [],
            "dev": payload.get("dev_task_ids") or [],
            "eval": payload.get("eval_task_ids") or [],
        },
        artifact_paths={
            "optimized_program_path": payload.get("optimized_program_path"),
            "summary_path": payload.get("summary_path"),
            "run_log_path": payload.get("run_log_path"),
            "event_log_path": payload.get("event_log_path"),
        },
        final_program_id=final_program_id,
    )


def program_reports(
    *,
    session_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
    run_id: str,
    final_program_id: str,
    parse_notes: list[str],
) -> list[GepaProgram]:
    reports = [
        GepaProgram(
            id=f"{run_id}:program:baseline",
            run_id=run_id,
            phase=ProgramPhase.BASELINE,
            source=source_ref(summary_path, session_dir, json_path="$.baseline_scores"),
            confidence=ExtractionConfidence.DEFINITELY_EXTRACTABLE,
        )
    ]
    program_path = resolve_artifact_path(
        session_dir,
        summary_path,
        summary.get("optimized_program_path"),
    )
    if program_path is None or not program_path.exists():
        parse_notes.append(f"missing optimized program for {summary_path}")
        return reports

    payload = read_json_file(program_path)
    complete = payload.get("complete") or {}
    signature = complete.get("signature") or {}
    reports.append(
        GepaProgram(
            id=final_program_id,
            run_id=run_id,
            phase=ProgramPhase.OPTIMIZED,
            predictor_name="complete" if "complete" in payload else None,
            source=source_ref(program_path, session_dir, json_path="$.complete"),
            signature_instructions=signature.get("instructions"),
            signature_fields=signature.get("fields") or [],
            demos=complete.get("demos") or [],
            train=complete.get("train") or [],
            traces=complete.get("traces") or [],
            lm=complete.get("lm"),
            dependency_versions=(payload.get("metadata") or {}).get(
                "dependency_versions",
                {},
            ),
            confidence=ExtractionConfidence.DEFINITELY_EXTRACTABLE,
        )
    )
    return reports


def split_score_reports(
    *,
    session_dir: Path,
    source_path: Path,
    summary: dict[str, Any],
    run_id: str,
    final_program_id: str,
) -> tuple[list[GepaSplitEvaluation], list[GepaTaskScore]]:
    split_reports: list[GepaSplitEvaluation] = []
    task_score_reports: list[GepaTaskScore] = []
    for phase, phase_key in (
        (ProgramPhase.BASELINE, "baseline_scores"),
        (ProgramPhase.OPTIMIZED, "optimized_scores"),
    ):
        for split, raw_score in (summary.get(phase_key) or {}).items():
            if not isinstance(raw_score, dict):
                continue
            evaluation_id = f"{run_id}:{phase.value}:{split}"
            program_id = (
                f"{run_id}:program:baseline"
                if phase == ProgramPhase.BASELINE
                else final_program_id
            )
            split_reports.append(
                GepaSplitEvaluation(
                    id=evaluation_id,
                    run_id=run_id,
                    phase=phase,
                    split=str(split),
                    label=str(raw_score.get("split_name") or f"{phase.value}/{split}"),
                    program_id=program_id,
                    task_count=int(raw_score.get("task_count") or 0),
                    average_pass_rate=float(raw_score.get("average_pass_rate") or 0.0),
                    full_pass_count=int(raw_score.get("full_pass_count") or 0),
                    full_pass_rate=float(raw_score.get("full_pass_rate") or 0.0),
                    source=source_ref(
                        source_path, session_dir, json_path=f"$.{phase_key}.{split}"
                    ),
                    confidence=ExtractionConfidence.DEFINITELY_EXTRACTABLE,
                )
            )
            task_scores = raw_score.get("task_scores") or {}
            for task_id, pass_rate in task_scores.items():
                task_score_reports.append(
                    GepaTaskScore(
                        id=f"{evaluation_id}:{task_id_to_slug(str(task_id))}",
                        run_id=run_id,
                        evaluation_id=evaluation_id,
                        phase=phase,
                        split=str(split),
                        task_id=str(task_id),
                        program_id=program_id,
                        valset_index=valset_index(summary, str(split), str(task_id)),
                        pass_rate=float(pass_rate),
                        source=source_ref(
                            source_path,
                            session_dir,
                            json_path=f"$.{phase_key}.{split}.task_scores.{task_id}",
                        ),
                        confidence=ExtractionConfidence.DEFINITELY_EXTRACTABLE,
                    )
                )
    return split_reports, task_score_reports


def metric_call_reports(
    *,
    session_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
    run_id: str,
    final_program_id: str,
    parse_notes: list[str],
) -> list[GepaMetricCall]:
    event_path = resolve_artifact_path(
        session_dir, summary_path, summary.get("event_log_path")
    )
    if event_path is None or not event_path.exists():
        parse_notes.append(f"missing event log for {summary_path}")
        return []

    reports: list[GepaMetricCall] = []
    eval_phase_ranges = eval_metric_call_ranges(summary)
    for line_number, line in enumerate(
        event_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = parse_json_text(line)
        except ValueError as exc:
            parse_notes.append(f"{event_path}:{line_number}: {exc}")
            continue
        event = record.get("event")
        if event not in {MetricCallKind.EVAL.value, MetricCallKind.GEPA.value}:
            continue
        payload = record.get("payload") or {}
        metric_call = payload.get("metric_call")
        pass_rate = payload.get("pass_rate")
        phase, split = infer_metric_phase_split(event, metric_call, eval_phase_ranges)
        kind = MetricCallKind(event)
        program_id = metric_call_program_id(run_id, final_program_id, phase)
        reports.append(
            GepaMetricCall(
                id=f"{run_id}:{event}:{len(reports):06d}",
                run_id=run_id,
                kind=kind,
                timestamp=record.get("timestamp"),
                label=payload.get("label"),
                metric_call=metric_call,
                phase=phase,
                split=split,
                program_id=program_id,
                predictor=payload.get("predictor"),
                task_id=payload.get("task_id"),
                pass_rate=None if pass_rate is None else float(pass_rate),
                error=payload.get("error"),
                source=source_ref(event_path, session_dir, line_number=line_number),
                confidence=(
                    ExtractionConfidence.EXTRACTABLE_WITH_INFERENCE
                    if phase is not None
                    else ExtractionConfidence.DEFINITELY_EXTRACTABLE
                ),
            )
        )
    return reports


def optimizer_iteration_reports(
    *,
    session_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
    run_id: str,
    parse_notes: list[str],
) -> list[GepaOptimizerIteration]:
    run_log_path = resolve_artifact_path(
        session_dir, summary_path, summary.get("run_log_path")
    )
    if run_log_path is None or not run_log_path.exists():
        parse_notes.append(f"missing run log for {summary_path}")
        return []

    lines = run_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    iteration_data: dict[int, dict[str, Any]] = {}
    parse_proposals(lines, iteration_data)
    parse_iteration_lines(lines, iteration_data, parse_notes, run_log_path)
    reports = []
    for iteration in sorted(iteration_data):
        data = iteration_data[iteration]
        reports.append(
            GepaOptimizerIteration(
                id=f"{run_id}:iteration:{iteration:06d}",
                run_id=run_id,
                iteration=iteration,
                selected_program_index=data.get("selected_program_index"),
                selected_score=data.get("selected_score"),
                proposed_predictor=data.get("proposed_predictor"),
                proposal_text=data.get("proposal_text"),
                proposal_char_count=(
                    len(data["proposal_text"]) if data.get("proposal_text") else None
                ),
                new_subsample_score=data.get("new_subsample_score"),
                old_subsample_score=data.get("old_subsample_score"),
                subsample_comparison=data.get("subsample_comparison"),
                subsample_action=data.get("subsample_action"),
                valset_score=data.get("valset_score"),
                valset_coverage=data.get("valset_coverage"),
                individual_valset_scores=data.get("individual_valset_scores", {}),
                pareto_front_scores=data.get("pareto_front_scores", {}),
                pareto_front_programs=data.get("pareto_front_programs", {}),
                pareto_aggregate_score=data.get("pareto_aggregate_score"),
                best_program_index=data.get("best_program_index"),
                best_score=data.get("best_score"),
                linear_pareto_program_index=data.get("linear_pareto_program_index"),
                new_candidate_index=data.get("new_candidate_index"),
                found_better_score=data.get("found_better_score"),
                skip_messages=data.get("skip_messages", []),
                source_lines=sorted(set(data.get("source_lines", []))),
            )
        )
    return reports


def parse_proposals(
    lines: list[str], iteration_data: dict[int, dict[str, Any]]
) -> None:
    current: tuple[int, str] | None = None
    current_lines: list[str] = []
    current_source_line: int | None = None
    for line_number, line in enumerate(lines, 1):
        match = PROPOSAL_RE.search(line)
        if match:
            if current is not None:
                save_proposal(
                    iteration_data,
                    current,
                    current_lines,
                    current_source_line,
                )
            current = (int(match.group("iteration")), match.group("predictor").strip())
            current_source_line = line_number
            current_lines = [match.group("text")]
            continue
        if current is None:
            continue
        if RUN_LOG_TIMESTAMP_RE.match(line):
            save_proposal(
                iteration_data,
                current,
                current_lines,
                current_source_line,
            )
            current = None
            current_lines = []
            current_source_line = None
            continue
        current_lines.append(line)
    if current is not None:
        save_proposal(iteration_data, current, current_lines, current_source_line)


def save_proposal(
    iteration_data: dict[int, dict[str, Any]],
    current: tuple[int, str],
    current_lines: list[str],
    current_source_line: int | None,
) -> None:
    iteration, predictor = current
    data = iteration_record(iteration_data, iteration)
    data["proposed_predictor"] = predictor
    data["proposal_text"] = "\n".join(current_lines).strip()
    if current_source_line is not None:
        data.setdefault("source_lines", []).append(current_source_line)


def parse_iteration_lines(
    lines: list[str],
    iteration_data: dict[int, dict[str, Any]],
    parse_notes: list[str],
    run_log_path: Path,
) -> None:
    patterns = [
        (
            re.compile(r"Iteration (\d+): Selected program (\d+) score: " + FLOAT_RE),
            lambda data, match: data.update(
                selected_program_index=int(match.group(2)),
                selected_score=float(match.group(3)),
            ),
        ),
        (
            re.compile(
                r"Iteration (\d+): New subsample score "
                + FLOAT_RE
                + r" is (better|not better) than old score "
                + FLOAT_RE
                + r"\.?(?:, (.*))?"
            ),
            lambda data, match: data.update(
                new_subsample_score=float(match.group(2)),
                subsample_comparison=match.group(3),
                old_subsample_score=float(match.group(4)),
                subsample_action=match.group(5),
            ),
        ),
        (
            re.compile(
                r"Iteration (\d+): Valset score for new program: "
                + FLOAT_RE
                + r" \(coverage (\d+) / (\d+)\)"
            ),
            lambda data, match: data.update(
                valset_score=float(match.group(2)),
                valset_coverage=f"{match.group(3)}/{match.group(4)}",
            ),
        ),
        (
            re.compile(
                r"Iteration (\d+): Found a better program on the valset with score "
                + FLOAT_RE
            ),
            lambda data, match: data.update(found_better_score=float(match.group(2))),
        ),
        (
            re.compile(
                r"Iteration (\d+): Valset pareto front aggregate score: " + FLOAT_RE
            ),
            lambda data, match: data.update(
                pareto_aggregate_score=float(match.group(2))
            ),
        ),
        (
            re.compile(
                r"Iteration (\d+): Best program as per aggregate score on valset: (\d+)"
            ),
            lambda data, match: data.update(best_program_index=int(match.group(2))),
        ),
        (
            re.compile(r"Iteration (\d+): Best score on valset: " + FLOAT_RE),
            lambda data, match: data.update(best_score=float(match.group(2))),
        ),
        (
            re.compile(r"Iteration (\d+): Linear pareto front program index: (\d+)"),
            lambda data, match: data.update(
                linear_pareto_program_index=int(match.group(2))
            ),
        ),
        (
            re.compile(r"Iteration (\d+): New program candidate index: (\d+)"),
            lambda data, match: data.update(new_candidate_index=int(match.group(2))),
        ),
    ]
    dict_patterns = [
        (
            re.compile(
                r"Iteration (\d+): Individual valset scores for new program: (\{.*\})"
            ),
            "individual_valset_scores",
            "float",
        ),
        (
            re.compile(r"Iteration (\d+): New valset pareto front scores: (\{.*\})"),
            "pareto_front_scores",
            "float",
        ),
        (
            re.compile(
                r"Iteration (\d+): Updated valset pareto front programs: (\{.*\})"
            ),
            "pareto_front_programs",
            "set_list",
        ),
    ]
    skip_patterns = [
        re.compile(
            r"Iteration (\d+): Reflective mutation did not propose a new candidate"
        ),
        re.compile(r"Iteration (\d+): All subsample scores perfect\. Skipping\."),
        re.compile(r"Iteration (\d+): No merge candidates found"),
    ]
    for line_number, line in enumerate(lines, 1):
        for pattern, updater in patterns:
            match = pattern.search(line)
            if match is None:
                continue
            data = iteration_record(iteration_data, int(match.group(1)))
            updater(data, match)
            data.setdefault("source_lines", []).append(line_number)
        for pattern, field_name, value_kind in dict_patterns:
            match = pattern.search(line)
            if match is None:
                continue
            data = iteration_record(iteration_data, int(match.group(1)))
            try:
                data[field_name] = parse_int_keyed_dict(match.group(2), value_kind)
            except (SyntaxError, ValueError) as exc:
                parse_notes.append(f"{run_log_path}:{line_number}: {exc}")
            data.setdefault("source_lines", []).append(line_number)
        for pattern in skip_patterns:
            match = pattern.search(line)
            if match is None:
                continue
            data = iteration_record(iteration_data, int(match.group(1)))
            data.setdefault("skip_messages", []).append(match.group(0))
            data.setdefault("source_lines", []).append(line_number)


def generated_output_reports(
    *,
    session_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
    run_id: str,
    final_program_id: str,
    parse_notes: list[str],
) -> list[GepaGeneratedOutput]:
    gepa_log_dir = infer_gepa_log_dir(session_dir, summary_path, summary)
    if gepa_log_dir is None:
        return []
    generated_dir = gepa_log_dir / GENERATED_BEST_OUTPUTS_DIR
    if not generated_dir.exists():
        return []
    dev_task_ids = summary.get("dev_task_ids") or []
    reports: list[GepaGeneratedOutput] = []
    for path in sorted(generated_dir.rglob("*.json")):
        indices = generated_output_indices(path)
        try:
            payload = read_json_file(path)
        except ValueError as exc:
            parse_notes.append(f"{path}: {exc}")
            continue
        valset_index = indices.get("task")
        task_id = (
            dev_task_ids[valset_index]
            if valset_index is not None and valset_index < len(dev_task_ids)
            else None
        )
        reports.append(
            GepaGeneratedOutput(
                id=f"{run_id}:generated_output:{len(reports):06d}",
                run_id=run_id,
                program_id=final_program_id,
                task_id=task_id,
                valset_index=valset_index,
                iteration=indices.get("iter"),
                program_index=indices.get("prog"),
                output_fields=payload,
                completed_code=payload.get("completed_code"),
                source=source_ref(path, session_dir),
                confidence=(
                    ExtractionConfidence.EXTRACTABLE_WITH_INFERENCE
                    if task_id is not None
                    else ExtractionConfidence.BEST_EFFORT
                ),
            )
        )
    return reports


def inspect_state_file(path: Path, session_dir: Path) -> GepaStateFile:
    size = path.stat().st_size
    sha256 = hash_file(path)
    opcode_counts: Counter[str] = Counter()
    string_keys: list[str] = []
    protocol: int | None = None
    parse_error = None
    try:
        for op, arg, _pos in pickletools.genops(path.read_bytes()):
            opcode_counts[op.name] += 1
            if op.name == "PROTO" and arg is not None:
                protocol = int(arg)
            if op.name.endswith("UNICODE") and isinstance(arg, str):
                string_keys.append(arg)
    except Exception as exc:  # pickletools raises several low-level parse errors.
        parse_error = f"{type(exc).__name__}: {exc}"
    unsafe = sorted(UNSAFE_PICKLE_OPS & set(opcode_counts))
    return GepaStateFile(
        source_file=str(path),
        source_relative_path=relative_to_session(path, session_dir),
        size_bytes=size,
        sha256=sha256,
        pickle_protocol=protocol,
        opcode_counts=dict(sorted(opcode_counts.items())),
        unsafe_opcode_names=unsafe,
        string_keys_seen=dedupe_preserving_order(string_keys),
        parse_error=parse_error,
    )


def candidate_rankings_for_iterations(
    iterations: list[GepaOptimizerIteration],
) -> list[GepaCandidateRanking]:
    rankings = []
    for iteration in iterations:
        if not any(
            value is not None
            for value in (
                iteration.new_candidate_index,
                iteration.valset_score,
                iteration.best_program_index,
                iteration.best_score,
                iteration.pareto_aggregate_score,
                iteration.linear_pareto_program_index,
            )
        ):
            continue
        rankings.append(
            GepaCandidateRanking(
                id=f"{iteration.run_id}:candidate_ranking:{iteration.iteration:06d}",
                run_id=iteration.run_id,
                iteration=iteration.iteration,
                candidate_index=iteration.new_candidate_index,
                valset_score=iteration.valset_score,
                best_program_index=iteration.best_program_index,
                best_score=iteration.best_score,
                pareto_aggregate_score=iteration.pareto_aggregate_score,
                linear_pareto_program_index=iteration.linear_pareto_program_index,
                source_lines=iteration.source_lines,
            )
        )
    return rankings


def candidate_program_reports(
    *,
    session_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
    iterations: list[GepaOptimizerIteration],
) -> list[GepaProgram]:
    run_log_path = resolve_artifact_path(
        session_dir,
        summary_path,
        summary.get("run_log_path"),
    )
    if run_log_path is None:
        run_log_path = summary_path

    reports = []
    for iteration in iterations:
        if iteration.proposal_text is None:
            continue
        candidate_index = iteration.new_candidate_index
        candidate_label = (
            f"{candidate_index:06d}"
            if candidate_index is not None
            else f"proposal_iteration_{iteration.iteration:06d}"
        )
        reports.append(
            GepaProgram(
                id=f"{iteration.run_id}:program:candidate:{candidate_label}",
                run_id=iteration.run_id,
                phase=ProgramPhase.CANDIDATE,
                candidate_index=candidate_index,
                predictor_name=iteration.proposed_predictor,
                parent_program_id=parent_program_id(iteration),
                source=source_ref(
                    run_log_path,
                    session_dir,
                    line_number=first_line_number(iteration.source_lines),
                ),
                signature_instructions=iteration.proposal_text,
                confidence=ExtractionConfidence.BEST_EFFORT,
            )
        )
    return reports


def candidate_valset_scores_for_iterations(
    iterations: list[GepaOptimizerIteration],
) -> list[dict[str, Any]]:
    rows = []
    for iteration in iterations:
        for valset_index, score in iteration.individual_valset_scores.items():
            rows.append(
                {
                    "run_id": iteration.run_id,
                    "iteration": iteration.iteration,
                    "candidate_index": iteration.new_candidate_index,
                    "valset_index": valset_index,
                    "pass_rate": score,
                    "source_lines": iteration.source_lines,
                    "confidence": ExtractionConfidence.BEST_EFFORT.value,
                }
            )
    return rows


def candidate_proposals_for_iterations(
    iterations: list[GepaOptimizerIteration],
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": iteration.run_id,
            "iteration": iteration.iteration,
            "candidate_index": iteration.new_candidate_index,
            "predictor": iteration.proposed_predictor,
            "proposal_text": iteration.proposal_text,
            "source_lines": iteration.source_lines,
            "confidence": ExtractionConfidence.BEST_EFFORT.value,
        }
        for iteration in iterations
        if iteration.proposal_text is not None
    ]


def parse_int_keyed_dict(value: str, value_kind: str) -> dict[int, Any]:
    raw = ast.literal_eval(value)
    if not isinstance(raw, dict):
        raise ValueError("expected dict literal")
    parsed: dict[int, Any] = {}
    for key, item in raw.items():
        int_key = int(key)
        if value_kind == "float":
            parsed[int_key] = float(item)
        elif value_kind == "set_list":
            parsed[int_key] = sorted(int(value) for value in item)
        else:
            parsed[int_key] = item
    return parsed


def infer_metric_phase_split(
    event: str,
    metric_call: Any,
    eval_phase_ranges: list[tuple[int, int, str, str]],
) -> tuple[str | None, str | None]:
    if event != MetricCallKind.EVAL.value or metric_call is None:
        return None, None
    metric_call_int = int(metric_call)
    for start, end, phase, split in eval_phase_ranges:
        if start <= metric_call_int <= end:
            return phase, split
    return None, None


def metric_call_program_id(
    run_id: str,
    final_program_id: str,
    phase: str | None,
) -> str | None:
    if phase == ProgramPhase.BASELINE.value:
        return f"{run_id}:program:baseline"
    if phase == ProgramPhase.OPTIMIZED.value:
        return final_program_id
    return None


def eval_metric_call_ranges(summary: dict[str, Any]) -> list[tuple[int, int, str, str]]:
    ranges = []
    start = 1
    for phase, phase_key in (
        (ProgramPhase.BASELINE.value, "baseline_scores"),
        (ProgramPhase.OPTIMIZED.value, "optimized_scores"),
    ):
        for split in ("train", "dev", "eval"):
            task_count = int(
                ((summary.get(phase_key) or {}).get(split) or {}).get("task_count")
                or len(summary.get(f"{split}_task_ids") or [])
            )
            if task_count == 0:
                continue
            end = start + task_count - 1
            ranges.append((start, end, phase, split))
            start = end + 1
    return ranges


def infer_gepa_log_dir(
    session_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
) -> Path | None:
    run_log_path = resolve_artifact_path(
        session_dir, summary_path, summary.get("run_log_path")
    )
    if run_log_path is not None and run_log_path.exists():
        match = re.search(
            r"log_dir=(\S*gepa_logs/\S+)",
            run_log_path.read_text(encoding="utf-8", errors="replace"),
        )
        if match is not None:
            candidate = resolve_artifact_path(session_dir, summary_path, match.group(1))
            if candidate is not None and candidate.exists():
                return candidate
    candidates = sorted(
        path.parent
        for path in (session_dir / RAW_DIR_NAME).rglob(GEPA_STATE_FILE_NAME)
        if GEPA_LOG_DIR_MARKER in path.parts
    )
    return candidates[0] if len(candidates) == 1 else None


def generated_output_indices(path: Path) -> dict[str, int]:
    indices: dict[str, int] = {}
    for part in path.parts:
        task_match = re.fullmatch(r"task_(\d+)", part)
        if task_match:
            indices["task"] = int(task_match.group(1))
    file_match = re.fullmatch(r"iter_(\d+)_prog_(\d+)\.json", path.name)
    if file_match:
        indices["iter"] = int(file_match.group(1))
        indices["prog"] = int(file_match.group(2))
    return indices


def valset_index(summary: dict[str, Any], split: str, task_id: str) -> int | None:
    if split != "dev":
        return None
    dev_task_ids = summary.get("dev_task_ids") or []
    try:
        return dev_task_ids.index(task_id)
    except ValueError:
        return None


def iter_gepa_summary_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(f"*{SUMMARY_SUFFIX}")
        if "gepa" in path.name and "optimized" in path.name
    )


def iter_session_dirs(corpus_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in corpus_dir.iterdir()
        if path.is_dir() and (path / METADATA_FILE_NAME).exists()
    )


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


def report_file_name(report: GepaSessionReport) -> str:
    return f"{report.session['session_id']}{REPORT_FILE_SUFFIX}"


def corpus_report_file(
    report: GepaSessionReport,
    session_dir: Path,
    report_path: Path,
) -> GepaCorpusReportFile:
    generation_types = sorted(
        {
            run.generation_type
            for run in report.optimizer_runs
            if run.generation_type is not None
        }
    )
    return GepaCorpusReportFile(
        session_id=str(report.session["session_id"]),
        session_dir=str(session_dir),
        report_file=str(report_path),
        optimizer_run_count=len(report.optimizer_runs),
        program_count=len(report.programs),
        metric_call_count=len(report.metric_calls),
        generated_output_count=len(report.generated_outputs),
        generation_types=generation_types,
    )


def resolve_artifact_path(
    session_dir: Path,
    summary_path: Path,
    value: Any,
) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [
        session_dir / RAW_DIR_NAME / path,
        session_dir / path,
        summary_path.parent / path,
        summary_path.parent / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def source_ref(
    path: Path,
    session_dir: Path,
    *,
    line_number: int | None = None,
    json_path: str | None = None,
) -> GepaSourceRef:
    return GepaSourceRef(
        source_file=str(path),
        source_relative_path=relative_to_session(path, session_dir),
        line_number=line_number,
        json_path=json_path,
    )


def iteration_record(
    iteration_data: dict[int, dict[str, Any]],
    iteration: int,
) -> dict[str, Any]:
    return iteration_data.setdefault(iteration, {})


def run_id_for_summary(path: Path) -> str:
    return path.name.removesuffix(SUMMARY_SUFFIX)


def task_id_to_slug(task_id: str) -> str:
    return task_id.replace("/", "_")


def parent_program_id(iteration: GepaOptimizerIteration) -> str | None:
    selected_index = iteration.selected_program_index
    if selected_index is None:
        return None
    if selected_index == 0:
        return f"{iteration.run_id}:program:baseline"
    return f"{iteration.run_id}:program:candidate:{selected_index:06d}"


def first_line_number(source_lines: list[int]) -> int | None:
    return min(source_lines) if source_lines else None


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


def relative_to_session(path: Path, session_dir: Path) -> str:
    try:
        return str(path.relative_to(session_dir))
    except ValueError:
        return str(path)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def absence_notes() -> list[str]:
    return [
        (
            "not_present: historical GEPA artifacts do not contain full per-testcase "
            "results; only pass_rate/error summaries are available"
        ),
        (
            "not_present: historical GEPA artifacts do not contain full generation "
            "messages, model responses, usage, or cost records"
        ),
        (
            "not_present: historical GEPA artifacts do not contain every rollout "
            "candidate output; generated_best_outputs_valset only preserves best "
            "validation outputs"
        ),
    ]


def format_rate(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    app()
