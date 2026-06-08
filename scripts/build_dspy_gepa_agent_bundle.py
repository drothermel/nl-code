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


SCHEMA_VERSION = "dspy_gepa_agent_bundle_v0"
MANIFEST_FILE_NAME = "manifest.json"
DEFAULT_REPORTS_DIR_NAME = "parsed_gepa_reports"
DEFAULT_DATA_REPORTS_DIR = Path(
    "/Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0/parsed_gepa_reports"
)
DEFAULT_BUNDLE_FILE_NAME = "gepa_optimization_agent_bundle.json"
STATE_PROMPT_SENTINEL = "prog_candidate_val_subscores"
BASELINE_CANDIDATE_INDEX = 0


class PromptVariantKind(StrEnum):
    BASELINE = "baseline"
    OPTIMIZED = "optimized"
    CANDIDATE_PROMOTED = "candidate_promoted"
    CANDIDATE_REJECTED = "candidate_rejected"


class EvaluationType(StrEnum):
    SPLIT = "split"
    TASK = "task"
    METRIC_CALL = "metric_call"
    CANDIDATE_VALSET_TASK = "candidate_valset_task"


class PromptVariantStatus(StrEnum):
    BASELINE = "baseline"
    FINAL = "final"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    MISSING = "missing"


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file: str
    source_relative_path: str | None = None
    line_number: int | None = None
    json_path: str | None = None


class BundleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    source_reports_dir: str
    source_manifest_file: str
    report_count: int
    included_session_count: int
    notes: list[str] = Field(default_factory=list)


class SessionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    session_dir: str
    session_kind: str | None = None
    source_report_file: str
    generation_types: list[str]
    run_ids: list[str]
    split_context: dict[str, list[str]] = Field(default_factory=dict)
    availability: dict[str, Any] = Field(default_factory=dict)
    optimizer_runs: list[dict[str, Any]] = Field(default_factory=list)
    parse_notes: list[str] = Field(default_factory=list)


class ValsetTaskScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valset_index: int
    task_id: str | None = None
    pass_rate: float


class PromptVariantPerformance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_program_index: int | None = None
    selected_score: float | None = None
    new_subsample_score: float | None = None
    old_subsample_score: float | None = None
    subsample_comparison: str | None = None
    subsample_action: str | None = None
    valset_score: float | None = None
    valset_coverage: str | None = None
    found_better_score: float | None = None
    best_program_index: int | None = None
    best_score: float | None = None
    pareto_aggregate_score: float | None = None
    linear_pareto_program_index: int | None = None
    skip_messages: list[str] = Field(default_factory=list)
    source_lines: list[int] = Field(default_factory=list)


class PromptVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    run_id: str
    generation_type: str | None = None
    optimization_target: str | None = None
    kind: PromptVariantKind
    status: PromptVariantStatus
    candidate_index: int | None = None
    iteration: int | None = None
    predictor: str | None = None
    parent_prompt_id: str | None = None
    prompt_text: str | None = None
    prompt_sha256: str | None = None
    prompt_char_count: int | None = None
    confidence: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    performance: PromptVariantPerformance | None = None
    per_valset_task_scores: list[ValsetTaskScore] = Field(default_factory=list)
    is_final_saved_prompt: bool = False


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    run_id: str
    evaluation_type: EvaluationType
    prompt_variant_id: str | None = None
    phase: str | None = None
    split: str | None = None
    task_id: str | None = None
    valset_index: int | None = None
    metric_call_kind: str | None = None
    label: str | None = None
    pass_rate: float | None = None
    average_pass_rate: float | None = None
    full_pass_rate: float | None = None
    task_count: int | None = None
    error: str | None = None
    confidence: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class GeneratedOutputBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    run_id: str
    prompt_variant_id: str | None = None
    task_id: str | None = None
    valset_index: int | None = None
    iteration: int | None = None
    program_index: int | None = None
    completed_code: str | None = None
    output_fields: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[SourceRef] = Field(default_factory=list)
    confidence: str | None = None


class StateSummaryBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    source_file: str
    source_relative_path: str
    size_bytes: int
    sha256: str | None = None
    decoded: bool
    pickle_protocol: int | None = None
    unsafe_opcode_names: list[str] = Field(default_factory=list)
    string_keys_seen: list[str] = Field(default_factory=list)
    parse_error: str | None = None


class StatePromptCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_index: int
    prompt_text: str
    source_ref: SourceRef


class GepaAgentBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    bundle_metadata: BundleMetadata
    sessions: list[SessionBundle]
    prompt_variants: list[PromptVariant]
    evaluations: list[EvaluationRecord]
    generated_outputs: list[GeneratedOutputBundle]
    state_summaries: list[StateSummaryBundle]
    agent_index: dict[str, Any]
    not_present: list[dict[str, Any]] = Field(default_factory=list)


app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(
        DEFAULT_DATA_REPORTS_DIR,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help=(
            "Parsed GEPA reports directory, or corpus root containing "
            "parsed_gepa_reports/."
        ),
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Write the agent bundle to this path.",
    ),
) -> None:
    reports_dir = resolve_reports_dir(input_path)
    output_path = output_file or default_output_path(input_path, reports_dir)
    bundle = build_agent_bundle(reports_dir)
    write_json_file(output_path, bundle.model_dump(mode="json"))
    render_bundle_summary(bundle, output_path)


def build_agent_bundle(reports_dir: Path) -> GepaAgentBundle:
    reports_dir = reports_dir.resolve()
    manifest_path = reports_dir / MANIFEST_FILE_NAME
    manifest = read_json_file(manifest_path)
    report_paths = report_paths_from_manifest(manifest, reports_dir)

    sessions: list[SessionBundle] = []
    prompt_variants: list[PromptVariant] = []
    evaluations: list[EvaluationRecord] = []
    generated_outputs: list[GeneratedOutputBundle] = []
    state_summaries: list[StateSummaryBundle] = []
    not_present: list[dict[str, Any]] = []

    for report_path in report_paths:
        report = read_json_file(report_path)
        context = session_context(report, report_path)
        sessions.append(session_bundle(report, report_path, context))
        session_prompt_variants = prompt_variants_for_report(report, context)
        prompt_variants.extend(session_prompt_variants)
        prompt_id_by_program_id = program_prompt_index(session_prompt_variants)
        evaluations.extend(
            evaluation_records_for_report(
                report,
                context,
                prompt_id_by_program_id,
                session_prompt_variants,
            )
        )
        generated_outputs.extend(
            generated_outputs_for_report(report, context, prompt_id_by_program_id)
        )
        state_summaries.extend(state_summaries_for_report(report, context))
        not_present.extend(not_present_records_for_report(report, context))

    metadata = BundleMetadata(
        created_at=datetime.now(timezone.utc),
        source_reports_dir=str(reports_dir),
        source_manifest_file=str(manifest_path),
        report_count=len(report_paths),
        included_session_count=len(sessions),
        notes=[
            "Derived from parsed GEPA forensic reports.",
            "Prompt text is preserved without truncation.",
            "State information comes from safe pickletools/string scans only.",
        ],
    )
    return GepaAgentBundle(
        bundle_metadata=metadata,
        sessions=sessions,
        prompt_variants=prompt_variants,
        evaluations=evaluations,
        generated_outputs=generated_outputs,
        state_summaries=state_summaries,
        agent_index=build_agent_index(
            sessions=sessions,
            prompt_variants=prompt_variants,
            evaluations=evaluations,
            generated_outputs=generated_outputs,
        ),
        not_present=not_present,
    )


def session_context(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    session = report.get("session") or {}
    session_id = str(session.get("session_id") or report_path.stem)
    runs = report.get("optimizer_runs") or []
    run = runs[0] if runs else {}
    split_context = run.get("split_task_ids") or {}
    dev_task_ids = split_context.get("dev") or []
    return {
        "session_id": session_id,
        "session_dir": str(session.get("session_dir") or ""),
        "report_path": str(report_path),
        "run": run,
        "run_id": str(run.get("id") or f"{session_id}:run:unknown"),
        "generation_type": run.get("generation_type"),
        "optimization_target": run.get("optimization_target"),
        "split_context": split_context,
        "dev_task_ids": dev_task_ids,
        "final_program_id": run.get("final_program_id"),
    }


def session_bundle(
    report: dict[str, Any],
    report_path: Path,
    context: dict[str, Any],
) -> SessionBundle:
    session = report.get("session") or {}
    optimizer_runs = report.get("optimizer_runs") or []
    generation_types = sorted(
        {
            run.get("generation_type")
            for run in optimizer_runs
            if run.get("generation_type") is not None
        }
    )
    return SessionBundle(
        session_id=context["session_id"],
        session_dir=context["session_dir"],
        session_kind=session.get("session_kind"),
        source_report_file=str(report_path),
        generation_types=generation_types,
        run_ids=[str(run.get("id")) for run in optimizer_runs if run.get("id")],
        split_context=context["split_context"],
        availability={
            "optimizer_run_count": len(optimizer_runs),
            "program_count": len(report.get("programs") or []),
            "metric_call_count": len(report.get("metric_calls") or []),
            "generated_output_count": len(report.get("generated_outputs") or []),
            "optimizer_iteration_count": len(report.get("optimizer_iterations") or []),
            "state_file_count": len(report.get("state_files") or []),
            "has_run_log": not any(
                "missing run log" in note for note in report.get("parse_notes") or []
            ),
            "has_event_log": not any(
                "missing event log" in note for note in report.get("parse_notes") or []
            ),
        },
        optimizer_runs=optimizer_runs,
        parse_notes=report.get("parse_notes") or [],
    )


def prompt_variants_for_report(
    report: dict[str, Any],
    context: dict[str, Any],
) -> list[PromptVariant]:
    state_prompt_records = state_program_candidate_prompt_records(report)
    state_prompts = {
        candidate_index: record.prompt_text
        for candidate_index, record in state_prompt_records.items()
    }
    program_by_id = {program["id"]: program for program in report.get("programs") or []}
    iterations = report.get("optimizer_iterations") or []
    iterations_by_candidate = {
        iteration.get("new_candidate_index"): iteration
        for iteration in iterations
        if iteration.get("new_candidate_index") is not None
    }
    final_prompt = final_optimized_prompt(report)

    variants = [
        baseline_prompt_variant(report, context, state_prompt_records),
        optimized_prompt_variant(report, context),
    ]

    promoted_candidate_indices = sorted(
        {
            int(iteration["new_candidate_index"])
            for iteration in iterations
            if iteration.get("new_candidate_index") is not None
        }
    )
    for candidate_index in promoted_candidate_indices:
        iteration = iterations_by_candidate[candidate_index]
        prompt_text = state_prompts.get(candidate_index)
        candidate_program = program_by_id.get(
            candidate_program_id(context["run_id"], candidate_index)
        )
        if prompt_text is None and candidate_program is not None:
            prompt_text = candidate_program.get("signature_instructions")
        variants.append(
            candidate_prompt_variant(
                context=context,
                iteration=iteration,
                prompt_text=prompt_text,
                kind=PromptVariantKind.CANDIDATE_PROMOTED,
                status=PromptVariantStatus.PROMOTED,
                candidate_index=candidate_index,
                is_final_saved_prompt=prompt_text is not None
                and prompt_text == final_prompt,
                source_refs=source_refs_for_candidate(
                    candidate_program,
                    iteration,
                    context,
                    state_prompt_records.get(candidate_index),
                ),
            )
        )

    for iteration in iterations:
        if not iteration.get("proposal_text"):
            continue
        if iteration.get("new_candidate_index") is not None:
            continue
        variants.append(
            candidate_prompt_variant(
                context=context,
                iteration=iteration,
                prompt_text=iteration.get("proposal_text"),
                kind=PromptVariantKind.CANDIDATE_REJECTED,
                status=PromptVariantStatus.REJECTED,
                candidate_index=None,
                is_final_saved_prompt=iteration.get("proposal_text") == final_prompt,
                source_refs=source_refs_from_iteration(iteration, context),
            )
        )

    return variants


def baseline_prompt_variant(
    report: dict[str, Any],
    context: dict[str, Any],
    state_prompt_records: dict[int, StatePromptCandidate],
) -> PromptVariant:
    baseline_program = first_program_by_phase(report, "baseline")
    state_prompt = state_prompt_records.get(BASELINE_CANDIDATE_INDEX)
    prompt_text = state_prompt.prompt_text if state_prompt is not None else None
    if prompt_text is None and baseline_program is not None:
        prompt_text = baseline_program.get("signature_instructions")
    return PromptVariant(
        id=baseline_prompt_id(context["session_id"], context["run_id"]),
        session_id=context["session_id"],
        run_id=context["run_id"],
        generation_type=context["generation_type"],
        optimization_target=context["optimization_target"],
        kind=PromptVariantKind.BASELINE,
        status=(
            PromptVariantStatus.BASELINE
            if prompt_text is not None
            else PromptVariantStatus.MISSING
        ),
        candidate_index=BASELINE_CANDIDATE_INDEX if prompt_text is not None else None,
        prompt_text=prompt_text,
        prompt_sha256=hash_text(prompt_text),
        prompt_char_count=len(prompt_text) if prompt_text is not None else None,
        confidence=(
            "extractable_with_inference" if prompt_text is not None else "not_present"
        ),
        source_refs=dedupe_source_refs(
            [
                *([state_prompt.source_ref] if state_prompt is not None else []),
                *source_refs_for_program(baseline_program),
            ]
        ),
        performance=baseline_performance(report),
        is_final_saved_prompt=False,
    )


def optimized_prompt_variant(
    report: dict[str, Any],
    context: dict[str, Any],
) -> PromptVariant:
    optimized_program = first_program_by_phase(report, "optimized")
    prompt_text = (
        optimized_program.get("signature_instructions")
        if optimized_program is not None
        else None
    )
    return PromptVariant(
        id=optimized_prompt_id(context["session_id"], context["run_id"]),
        session_id=context["session_id"],
        run_id=context["run_id"],
        generation_type=context["generation_type"],
        optimization_target=context["optimization_target"],
        kind=PromptVariantKind.OPTIMIZED,
        status=PromptVariantStatus.FINAL,
        prompt_text=prompt_text,
        prompt_sha256=hash_text(prompt_text),
        prompt_char_count=len(prompt_text) if prompt_text is not None else None,
        confidence=(
            optimized_program.get("confidence")
            if optimized_program is not None
            else "not_present"
        ),
        source_refs=source_refs_for_program(optimized_program),
        performance=optimized_performance(report),
        is_final_saved_prompt=True,
    )


def candidate_prompt_variant(
    *,
    context: dict[str, Any],
    iteration: dict[str, Any],
    prompt_text: str | None,
    kind: PromptVariantKind,
    status: PromptVariantStatus,
    candidate_index: int | None,
    is_final_saved_prompt: bool,
    source_refs: list[SourceRef],
) -> PromptVariant:
    prompt_id = (
        candidate_prompt_id(context["session_id"], context["run_id"], candidate_index)
        if candidate_index is not None
        else proposal_prompt_id(
            context["session_id"],
            context["run_id"],
            int(iteration["iteration"]),
        )
    )
    return PromptVariant(
        id=prompt_id,
        session_id=context["session_id"],
        run_id=context["run_id"],
        generation_type=context["generation_type"],
        optimization_target=context["optimization_target"],
        kind=kind,
        status=status,
        candidate_index=candidate_index,
        iteration=iteration.get("iteration"),
        predictor=iteration.get("proposed_predictor"),
        parent_prompt_id=parent_prompt_id(context, iteration),
        prompt_text=prompt_text,
        prompt_sha256=hash_text(prompt_text),
        prompt_char_count=len(prompt_text) if prompt_text is not None else None,
        confidence="best_effort",
        source_refs=source_refs,
        performance=performance_from_iteration(iteration),
        per_valset_task_scores=valset_task_scores_from_iteration(
            iteration,
            context["dev_task_ids"],
        ),
        is_final_saved_prompt=is_final_saved_prompt,
    )


def evaluation_records_for_report(
    report: dict[str, Any],
    context: dict[str, Any],
    prompt_id_by_program_id: dict[str, str],
    prompt_variants: list[PromptVariant],
) -> list[EvaluationRecord]:
    evaluations = []
    for split_eval in report.get("split_evaluations") or []:
        evaluations.append(
            split_evaluation_record(split_eval, context, prompt_id_by_program_id)
        )
    for task_score in report.get("task_scores") or []:
        evaluations.append(
            task_score_record(task_score, context, prompt_id_by_program_id)
        )
    for metric_call in report.get("metric_calls") or []:
        evaluations.append(
            metric_call_record(metric_call, context, prompt_id_by_program_id)
        )
    for prompt_variant in prompt_variants:
        for valset_score in prompt_variant.per_valset_task_scores:
            evaluations.append(
                EvaluationRecord(
                    id=(
                        f"{prompt_variant.id}:candidate_valset_task:"
                        f"{valset_score.valset_index:06d}"
                    ),
                    session_id=context["session_id"],
                    run_id=context["run_id"],
                    evaluation_type=EvaluationType.CANDIDATE_VALSET_TASK,
                    prompt_variant_id=prompt_variant.id,
                    phase="candidate",
                    split="dev",
                    task_id=valset_score.task_id,
                    valset_index=valset_score.valset_index,
                    pass_rate=valset_score.pass_rate,
                    confidence=prompt_variant.confidence,
                    source_refs=prompt_variant.source_refs,
                    details={
                        "iteration": prompt_variant.iteration,
                        "candidate_index": prompt_variant.candidate_index,
                    },
                )
            )
    return evaluations


def split_evaluation_record(
    split_eval: dict[str, Any],
    context: dict[str, Any],
    prompt_id_by_program_id: dict[str, str],
) -> EvaluationRecord:
    return EvaluationRecord(
        id=f"{context['session_id']}:{split_eval['id']}",
        session_id=context["session_id"],
        run_id=context["run_id"],
        evaluation_type=EvaluationType.SPLIT,
        prompt_variant_id=prompt_id_by_program_id.get(
            str(split_eval.get("program_id"))
        ),
        phase=split_eval.get("phase"),
        split=split_eval.get("split"),
        label=split_eval.get("label"),
        average_pass_rate=split_eval.get("average_pass_rate"),
        full_pass_rate=split_eval.get("full_pass_rate"),
        task_count=split_eval.get("task_count"),
        confidence=split_eval.get("confidence"),
        source_refs=source_ref_list(split_eval.get("source")),
    )


def task_score_record(
    task_score: dict[str, Any],
    context: dict[str, Any],
    prompt_id_by_program_id: dict[str, str],
) -> EvaluationRecord:
    return EvaluationRecord(
        id=f"{context['session_id']}:{task_score['id']}",
        session_id=context["session_id"],
        run_id=context["run_id"],
        evaluation_type=EvaluationType.TASK,
        prompt_variant_id=prompt_id_by_program_id.get(
            str(task_score.get("program_id"))
        ),
        phase=task_score.get("phase"),
        split=task_score.get("split"),
        task_id=task_score.get("task_id"),
        valset_index=task_score.get("valset_index"),
        pass_rate=task_score.get("pass_rate"),
        error=task_score.get("error"),
        confidence=task_score.get("confidence"),
        source_refs=source_ref_list(task_score.get("source")),
    )


def metric_call_record(
    metric_call: dict[str, Any],
    context: dict[str, Any],
    prompt_id_by_program_id: dict[str, str],
) -> EvaluationRecord:
    return EvaluationRecord(
        id=f"{context['session_id']}:{metric_call['id']}",
        session_id=context["session_id"],
        run_id=context["run_id"],
        evaluation_type=EvaluationType.METRIC_CALL,
        prompt_variant_id=prompt_id_by_program_id.get(
            str(metric_call.get("program_id"))
        ),
        phase=metric_call.get("phase"),
        split=metric_call.get("split"),
        task_id=metric_call.get("task_id"),
        metric_call_kind=metric_call.get("kind"),
        label=metric_call.get("label"),
        pass_rate=metric_call.get("pass_rate"),
        error=metric_call.get("error"),
        confidence=metric_call.get("confidence"),
        source_refs=source_ref_list(metric_call.get("source")),
        details={
            "timestamp": metric_call.get("timestamp"),
            "metric_call": metric_call.get("metric_call"),
            "predictor": metric_call.get("predictor"),
            "iteration": metric_call.get("iteration"),
        },
    )


def generated_outputs_for_report(
    report: dict[str, Any],
    context: dict[str, Any],
    prompt_id_by_program_id: dict[str, str],
) -> list[GeneratedOutputBundle]:
    outputs = []
    for item in report.get("generated_outputs") or []:
        outputs.append(
            GeneratedOutputBundle(
                id=f"{context['session_id']}:{item['id']}",
                session_id=context["session_id"],
                run_id=context["run_id"],
                prompt_variant_id=prompt_id_by_program_id.get(
                    str(item.get("program_id"))
                ),
                task_id=item.get("task_id"),
                valset_index=item.get("valset_index"),
                iteration=item.get("iteration"),
                program_index=item.get("program_index"),
                completed_code=item.get("completed_code"),
                output_fields=item.get("output_fields") or {},
                source_refs=source_ref_list(item.get("source")),
                confidence=item.get("confidence"),
            )
        )
    return outputs


def state_summaries_for_report(
    report: dict[str, Any],
    context: dict[str, Any],
) -> list[StateSummaryBundle]:
    summaries = []
    for index, state_file in enumerate(report.get("state_files") or []):
        summaries.append(
            StateSummaryBundle(
                id=f"{context['session_id']}:state:{index:06d}",
                session_id=context["session_id"],
                source_file=state_file.get("source_file"),
                source_relative_path=state_file.get("source_relative_path"),
                size_bytes=state_file.get("size_bytes"),
                sha256=state_file.get("sha256"),
                decoded=bool(state_file.get("decoded")),
                pickle_protocol=state_file.get("pickle_protocol"),
                unsafe_opcode_names=state_file.get("unsafe_opcode_names") or [],
                string_keys_seen=state_file.get("string_keys_seen") or [],
                parse_error=state_file.get("parse_error"),
            )
        )
    return summaries


def not_present_records_for_report(
    report: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for note in report.get("parse_notes") or []:
        if not ("not_present:" in note or note.startswith("missing ")):
            continue
        records.append(
            {
                "session_id": context["session_id"],
                "run_id": context["run_id"],
                "note": note,
            }
        )
    baseline = baseline_prompt_id(context["session_id"], context["run_id"])
    if not any(
        prompt.get("signature_instructions")
        for prompt in report.get("programs") or []
        if prompt.get("phase") == "baseline"
    ) and BASELINE_CANDIDATE_INDEX not in state_program_candidate_prompts(report):
        records.append(
            {
                "session_id": context["session_id"],
                "run_id": context["run_id"],
                "prompt_variant_id": baseline,
                "note": "not_present: baseline prompt text is not recoverable",
            }
        )
    return records


def state_program_candidate_prompts(report: dict[str, Any]) -> dict[int, str]:
    return {
        candidate_index: record.prompt_text
        for candidate_index, record in state_program_candidate_prompt_records(
            report
        ).items()
    }


def state_program_candidate_prompt_records(
    report: dict[str, Any],
) -> dict[int, StatePromptCandidate]:
    records: dict[int, StatePromptCandidate] = {}
    for state_file in report.get("state_files") or []:
        strings = state_file.get("string_keys_seen") or []
        if "program_candidates" not in strings:
            continue
        start = strings.index("program_candidates") + 1
        if start < len(strings) and looks_like_predictor_name(strings[start]):
            start += 1
        end = (
            strings.index(STATE_PROMPT_SENTINEL)
            if STATE_PROMPT_SENTINEL in strings
            else len(strings)
        )
        candidate_strings = [
            value for value in strings[start:end] if looks_like_prompt_text(value)
        ]
        for candidate_index, prompt in enumerate(candidate_strings):
            if candidate_index in records:
                continue
            records[candidate_index] = StatePromptCandidate(
                candidate_index=candidate_index,
                prompt_text=prompt,
                source_ref=SourceRef(
                    source_file=str(state_file.get("source_file") or ""),
                    source_relative_path=state_file.get("source_relative_path"),
                    json_path="state_files[].string_keys_seen",
                ),
            )
    return records


def baseline_performance(report: dict[str, Any]) -> PromptVariantPerformance | None:
    baseline_dev = split_eval_by_phase_split(report, "baseline", "dev")
    if baseline_dev is None:
        return None
    return PromptVariantPerformance(
        valset_score=baseline_dev.get("average_pass_rate"),
        valset_coverage=f"{baseline_dev.get('task_count')}/{baseline_dev.get('task_count')}",
    )


def optimized_performance(report: dict[str, Any]) -> PromptVariantPerformance | None:
    optimized_dev = split_eval_by_phase_split(report, "optimized", "dev")
    if optimized_dev is None:
        return None
    return PromptVariantPerformance(
        valset_score=optimized_dev.get("average_pass_rate"),
        valset_coverage=f"{optimized_dev.get('task_count')}/{optimized_dev.get('task_count')}",
    )


def performance_from_iteration(iteration: dict[str, Any]) -> PromptVariantPerformance:
    return PromptVariantPerformance(
        selected_program_index=iteration.get("selected_program_index"),
        selected_score=iteration.get("selected_score"),
        new_subsample_score=iteration.get("new_subsample_score"),
        old_subsample_score=iteration.get("old_subsample_score"),
        subsample_comparison=iteration.get("subsample_comparison"),
        subsample_action=iteration.get("subsample_action"),
        valset_score=iteration.get("valset_score"),
        valset_coverage=iteration.get("valset_coverage"),
        found_better_score=iteration.get("found_better_score"),
        best_program_index=iteration.get("best_program_index"),
        best_score=iteration.get("best_score"),
        pareto_aggregate_score=iteration.get("pareto_aggregate_score"),
        linear_pareto_program_index=iteration.get("linear_pareto_program_index"),
        skip_messages=iteration.get("skip_messages") or [],
        source_lines=iteration.get("source_lines") or [],
    )


def valset_task_scores_from_iteration(
    iteration: dict[str, Any],
    dev_task_ids: list[str],
) -> list[ValsetTaskScore]:
    scores = []
    raw_scores = iteration.get("individual_valset_scores") or {}
    for raw_index, pass_rate in sorted(
        raw_scores.items(),
        key=lambda item: int(item[0]),
    ):
        valset_index = int(raw_index)
        task_id = (
            dev_task_ids[valset_index] if valset_index < len(dev_task_ids) else None
        )
        scores.append(
            ValsetTaskScore(
                valset_index=valset_index,
                task_id=task_id,
                pass_rate=float(pass_rate),
            )
        )
    return scores


def program_prompt_index(prompt_variants: list[PromptVariant]) -> dict[str, str]:
    index = {}
    for prompt_variant in prompt_variants:
        if prompt_variant.kind == PromptVariantKind.BASELINE:
            index[f"{prompt_variant.run_id}:program:baseline"] = prompt_variant.id
        elif prompt_variant.kind == PromptVariantKind.OPTIMIZED:
            index[f"{prompt_variant.run_id}:program:optimized"] = prompt_variant.id
        elif (
            prompt_variant.kind == PromptVariantKind.CANDIDATE_PROMOTED
            and prompt_variant.candidate_index is not None
        ):
            index[
                candidate_program_id(
                    prompt_variant.run_id, prompt_variant.candidate_index
                )
            ] = prompt_variant.id
    return index


def build_agent_index(
    *,
    sessions: list[SessionBundle],
    prompt_variants: list[PromptVariant],
    evaluations: list[EvaluationRecord],
    generated_outputs: list[GeneratedOutputBundle],
) -> dict[str, Any]:
    prompt_by_session: dict[str, list[str]] = defaultdict(list)
    prompt_by_kind: dict[str, list[str]] = defaultdict(list)
    final_prompt_by_session: dict[str, str] = {}
    final_saved_prompts_by_session: dict[str, list[str]] = defaultdict(list)
    by_generation_type: dict[str, list[str]] = defaultdict(list)
    evaluations_by_task: dict[str, list[str]] = defaultdict(list)
    generated_outputs_by_task: dict[str, list[str]] = defaultdict(list)

    for session in sessions:
        for generation_type in session.generation_types:
            by_generation_type[generation_type].append(session.session_id)
    for prompt_variant in prompt_variants:
        prompt_by_session[prompt_variant.session_id].append(prompt_variant.id)
        prompt_by_kind[prompt_variant.kind.value].append(prompt_variant.id)
        if prompt_variant.is_final_saved_prompt:
            final_saved_prompts_by_session[prompt_variant.session_id].append(
                prompt_variant.id
            )
        if prompt_variant.kind == PromptVariantKind.OPTIMIZED:
            final_prompt_by_session[prompt_variant.session_id] = prompt_variant.id
    for evaluation in evaluations:
        if evaluation.task_id is not None:
            evaluations_by_task[evaluation.task_id].append(evaluation.id)
    for generated_output in generated_outputs:
        if generated_output.task_id is not None:
            generated_outputs_by_task[generated_output.task_id].append(
                generated_output.id
            )

    return {
        "session_ids": [session.session_id for session in sessions],
        "sessions_by_generation_type": sort_index(by_generation_type),
        "prompt_variants_by_session": sort_index(prompt_by_session),
        "prompt_variants_by_kind": sort_index(prompt_by_kind),
        "final_prompt_variant_by_session": dict(
            sorted(final_prompt_by_session.items())
        ),
        "final_saved_prompt_variants_by_session": sort_index(
            final_saved_prompts_by_session
        ),
        "evaluations_by_task_id": sort_index(evaluations_by_task),
        "generated_outputs_by_task_id": sort_index(generated_outputs_by_task),
    }


def resolve_reports_dir(input_path: Path) -> Path:
    input_path = input_path.resolve()
    if (input_path / MANIFEST_FILE_NAME).exists():
        return input_path
    candidate = input_path / DEFAULT_REPORTS_DIR_NAME
    if (candidate / MANIFEST_FILE_NAME).exists():
        return candidate
    raise typer.BadParameter(
        f"{input_path} must be a parsed GEPA reports directory or contain "
        f"{DEFAULT_REPORTS_DIR_NAME}/{MANIFEST_FILE_NAME}"
    )


def default_output_path(input_path: Path, reports_dir: Path) -> Path:
    if input_path.resolve() == reports_dir.resolve():
        return reports_dir.parent / DEFAULT_BUNDLE_FILE_NAME
    return input_path.resolve() / DEFAULT_BUNDLE_FILE_NAME


def report_paths_from_manifest(
    manifest: dict[str, Any],
    reports_dir: Path,
) -> list[Path]:
    paths = []
    for item in manifest.get("reports") or []:
        report_file = Path(str(item.get("report_file")))
        if not report_file.is_absolute():
            report_file = reports_dir / report_file
        paths.append(report_file)
    return sorted(paths)


def first_program_by_phase(report: dict[str, Any], phase: str) -> dict[str, Any] | None:
    return next(
        (
            program
            for program in report.get("programs") or []
            if program.get("phase") == phase
        ),
        None,
    )


def split_eval_by_phase_split(
    report: dict[str, Any],
    phase: str,
    split: str,
) -> dict[str, Any] | None:
    return next(
        (
            split_eval
            for split_eval in report.get("split_evaluations") or []
            if split_eval.get("phase") == phase and split_eval.get("split") == split
        ),
        None,
    )


def final_optimized_prompt(report: dict[str, Any]) -> str | None:
    optimized_program = first_program_by_phase(report, "optimized")
    if optimized_program is None:
        return None
    return optimized_program.get("signature_instructions")


def parent_prompt_id(context: dict[str, Any], iteration: dict[str, Any]) -> str | None:
    selected_index = iteration.get("selected_program_index")
    if selected_index is None:
        return None
    selected_index = int(selected_index)
    if selected_index == BASELINE_CANDIDATE_INDEX:
        return baseline_prompt_id(context["session_id"], context["run_id"])
    return candidate_prompt_id(context["session_id"], context["run_id"], selected_index)


def source_refs_for_program(program: dict[str, Any] | None) -> list[SourceRef]:
    if program is None:
        return []
    return source_ref_list(program.get("source"))


def source_refs_for_candidate(
    candidate_program: dict[str, Any] | None,
    iteration: dict[str, Any],
    context: dict[str, Any],
    state_prompt: StatePromptCandidate | None,
) -> list[SourceRef]:
    return dedupe_source_refs(
        [
            *([state_prompt.source_ref] if state_prompt is not None else []),
            *source_refs_for_program(candidate_program),
            *source_refs_from_iteration(iteration, context),
        ]
    )


def source_refs_from_iteration(
    iteration: dict[str, Any],
    context: dict[str, Any],
) -> list[SourceRef]:
    source_lines = iteration.get("source_lines") or []
    source_file = ""
    if context:
        source_file = (
            context.get("run", {}).get("artifact_paths", {}).get("run_log_path") or ""
        )
    return [
        SourceRef(
            source_file=source_file,
            source_relative_path=source_file,
            line_number=min(source_lines) if source_lines else None,
        )
    ]


def source_ref_list(source: dict[str, Any] | None) -> list[SourceRef]:
    if not source:
        return []
    return [
        SourceRef(
            source_file=str(source.get("source_file") or ""),
            source_relative_path=source.get("source_relative_path"),
            line_number=source.get("line_number"),
            json_path=source.get("json_path"),
        )
    ]


def dedupe_source_refs(source_refs: list[SourceRef]) -> list[SourceRef]:
    deduped = {}
    for source_ref in source_refs:
        key = (
            source_ref.source_file,
            source_ref.source_relative_path,
            source_ref.line_number,
            source_ref.json_path,
        )
        deduped[key] = source_ref
    return list(deduped.values())


def candidate_program_id(run_id: str, candidate_index: int) -> str:
    return f"{run_id}:program:candidate:{candidate_index:06d}"


def baseline_prompt_id(session_id: str, run_id: str) -> str:
    return f"{session_id}:{run_id}:prompt:baseline"


def optimized_prompt_id(session_id: str, run_id: str) -> str:
    return f"{session_id}:{run_id}:prompt:optimized"


def candidate_prompt_id(session_id: str, run_id: str, candidate_index: int) -> str:
    return f"{session_id}:{run_id}:prompt:candidate:{candidate_index:06d}"


def proposal_prompt_id(session_id: str, run_id: str, iteration: int) -> str:
    return f"{session_id}:{run_id}:prompt:proposal_iteration:{iteration:06d}"


def looks_like_predictor_name(value: str) -> bool:
    return len(value) < 80 and "\n" not in value and " " not in value


def looks_like_prompt_text(value: str) -> bool:
    return len(value) > 40 and (" " in value or "\n" in value)


def hash_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sort_index(index: dict[str, list[str]]) -> dict[str, list[str]]:
    return {key: sorted(values) for key, values in sorted(index.items())}


def read_json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def write_json_file(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def render_bundle_summary(bundle: GepaAgentBundle, output_path: Path) -> None:
    console = Console()
    console.print(
        Panel(
            "\n".join(
                [
                    f"Output: {output_path}",
                    f"Sessions: {len(bundle.sessions)}",
                    f"Prompt variants: {len(bundle.prompt_variants)}",
                    f"Evaluations: {len(bundle.evaluations)}",
                    f"Generated outputs: {len(bundle.generated_outputs)}",
                    f"State summaries: {len(bundle.state_summaries)}",
                ]
            ),
            title="GEPA Agent Bundle",
        )
    )
    table = Table(title="Sessions")
    table.add_column("Session")
    table.add_column("Types")
    table.add_column("Prompts", justify="right")
    table.add_column("Evaluations", justify="right")
    for session in bundle.sessions:
        prompt_count = sum(
            prompt.session_id == session.session_id for prompt in bundle.prompt_variants
        )
        eval_count = sum(
            evaluation.session_id == session.session_id
            for evaluation in bundle.evaluations
        )
        table.add_row(
            session.session_id,
            ", ".join(session.generation_types),
            str(prompt_count),
            str(eval_count),
        )
    console.print(table)


if __name__ == "__main__":
    app()
