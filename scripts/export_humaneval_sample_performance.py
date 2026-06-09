from __future__ import annotations

import csv
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, ConfigDict


DEFAULT_CORPUS_ROOT = Path(
    "/Users/daniellerothermel/drotherm/data/code-comp/dspy-exps/v0"
)
DEFAULT_EVIDENCE_FILE_NAME = "humaneval_sample_performance_evidence.csv"
DEFAULT_AGGREGATE_FILE_NAME = "humaneval_sample_performance_aggregate.csv"
DEFAULT_DIRECT_GPT5_NANO_FILE_NAME = "humaneval_gpt5nano_direct_sample_performance.csv"
DEFAULT_ENCDEC_GPT5_NANO_FILE_NAME = "humaneval_gpt5nano_encdec_sample_performance.csv"
DEFAULT_BASELINE_GPT5_NANO_FILE_NAME = (
    "humaneval_gpt5nano_baseline_sample_performance.csv"
)
DEFAULT_NON_BASELINE_GPT5_NANO_FILE_NAME = (
    "humaneval_gpt5nano_non_baseline_sample_performance.csv"
)
DEFAULT_ALL_SETTINGS_GPT5_NANO_FILE_NAME = (
    "humaneval_gpt5nano_all_settings_sample_performance.csv"
)
DEFAULT_WORST_SELECTIONS_GPT5_NANO_FILE_NAME = (
    "humaneval_gpt5nano_worst_all_settings_sample_sets.json"
)

BASELINE_BUCKET = "baseline"
NON_BASELINE_BUCKET = "non-baseline"
UNKNOWN_MODEL = "<unknown>"
GPT5_NANO_MODEL_NAME = "openrouter/openai/gpt-5-nano"
WORST_SELECTION_SIZES = (25, 50, 100)

OPTIMIZED_PROGRAM_PATH_KEYS = (
    "direct_program_path",
    "encdec_program_path",
    "encoder_program_path",
    "decoder_program_path",
)

EVIDENCE_FIELD_NAMES = (
    "direct or enc-dec",
    "sample id",
    "model name",
    "bucket",
    "pass rate",
    "source kind",
    "session id",
    "run id",
    "source record id",
    "source relative path",
    "phase",
    "split",
    "generation type",
    "optimization target",
)

AGGREGATE_FIELD_NAMES = (
    "direct or enc-dec",
    "sample id",
    "model name",
    "baseline perf",
    "baseline variance",
    "n baseline",
    "non-baseline perf",
    "non-baseline variance",
    "n non-baseline",
    "all perf",
    "all variance",
    "n all",
)

SAMPLE_SUMMARY_FIELD_NAMES = (
    "sample id",
    "average perf",
    "perf variance",
    "n average",
)


class SourceKind(StrEnum):
    EVAL_ATTEMPT = "eval_attempt"
    OPTIMIZER_SUMMARY_TASK_SCORE = "optimizer_summary_task_score"
    GEPA_SEARCH_METRIC_CALL = "gepa_search_metric_call"
    MIPRO_SEARCH_METRIC_CALL = "mipro_search_metric_call"


class EvidenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    sample_id: str
    model_name: str
    bucket: str
    pass_rate: float
    source_kind: SourceKind
    session_id: str
    run_id: str
    source_record_id: str
    source_relative_path: str
    phase: str
    split: str
    generation_type: str
    optimization_target: str

    def to_csv_row(self) -> dict[str, str | float]:
        return {
            "direct or enc-dec": self.family,
            "sample id": self.sample_id,
            "model name": self.model_name,
            "bucket": self.bucket,
            "pass rate": self.pass_rate,
            "source kind": self.source_kind.value,
            "session id": self.session_id,
            "run id": self.run_id,
            "source record id": self.source_record_id,
            "source relative path": self.source_relative_path,
            "phase": self.phase,
            "split": self.split,
            "generation type": self.generation_type,
            "optimization target": self.optimization_target,
        }


class AggregateValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_values: list[float]
    non_baseline_values: list[float]

    def to_csv_row(
        self, *, family: str, sample_id: str, model_name: str
    ) -> dict[str, str | int]:
        all_values = self.baseline_values + self.non_baseline_values
        return {
            "direct or enc-dec": family,
            "sample id": sample_id,
            "model name": model_name,
            "baseline perf": mean_or_blank(self.baseline_values),
            "baseline variance": variance_or_blank(self.baseline_values),
            "n baseline": len(self.baseline_values),
            "non-baseline perf": mean_or_blank(self.non_baseline_values),
            "non-baseline variance": variance_or_blank(self.non_baseline_values),
            "n non-baseline": len(self.non_baseline_values),
            "all perf": mean_or_blank(all_values),
            "all variance": variance_or_blank(all_values),
            "n all": len(all_values),
        }


def read_json_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def iter_jsonl(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def mean_or_blank(values: list[float]) -> str:
    mean = mean_or_none(values)
    if mean is None:
        return ""
    return f"{mean:.12g}"


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def variance_or_blank(values: list[float]) -> str:
    variance = variance_or_none(values)
    if variance is None:
        return ""
    return f"{variance:.12g}"


def variance_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def normalize_family(generation_type: str | None) -> str:
    if not generation_type:
        return "<unknown>"
    if generation_type == "encdec" or generation_type.startswith("encdec_"):
        return "enc-dec"
    if generation_type == "direct" or generation_type.startswith("direct_"):
        return "direct"
    return generation_type


def session_id_from_path(path: Path, corpus_root: Path) -> str:
    relative_parts = path.relative_to(corpus_root).parts
    if not relative_parts:
        return ""
    return relative_parts[0]


def source_relative_path(path: Path, corpus_root: Path) -> str:
    return str(path.relative_to(corpus_root))


def task_sort_key(task_id: str) -> tuple[str, int, str]:
    prefix, separator, suffix = task_id.partition("/")
    if separator and suffix.isdigit():
        return prefix, int(suffix), ""
    return task_id, -1, ""


def output_path_or_default(
    path: Path | None, corpus_root: Path, file_name: str
) -> Path:
    return path if path is not None else corpus_root / file_name


def eval_run_is_baseline(run: dict[str, Any]) -> bool:
    config = run.get("config") or {}
    if any(config.get(key) for key in OPTIMIZED_PROGRAM_PATH_KEYS):
        return False
    source_path = run.get("source_relative_path") or ""
    if "/mipro_" in source_path or "/gepa_" in source_path:
        return False
    return True


def model_name_for_attempt(
    attempt: dict[str, Any],
    run: dict[str, Any],
    generation_calls_by_id: dict[str, dict[str, Any]],
    report_default_model: str,
) -> str:
    config = run.get("config") or {}
    if config.get("model"):
        return config["model"]

    model_names = sorted(set(run.get("model_names") or []))
    if model_names:
        return "+".join(model_names)

    call_models = sorted(
        {
            generation_calls_by_id[call_id].get("model")
            for call_id in attempt.get("generation_call_ids") or []
            if call_id in generation_calls_by_id
            and generation_calls_by_id[call_id].get("model")
        }
    )
    if call_models:
        return "+".join(call_models)

    if report_default_model:
        return report_default_model

    return UNKNOWN_MODEL


def default_model_from_generation_calls(generation_calls: list[dict[str, Any]]) -> str:
    model_names = sorted(
        {
            generation_call.get("model")
            for generation_call in generation_calls
            if generation_call.get("model")
        }
    )
    if not model_names:
        return ""
    return "+".join(model_names)


def iter_eval_attempt_evidence(corpus_root: Path) -> list[EvidenceRow]:
    report_dir = corpus_root / "parsed_eval_reports"
    rows: list[EvidenceRow] = []
    for report_path in sorted(report_dir.glob("*.eval_report.json")):
        report = read_json_file(report_path)
        session = report.get("session") or {}
        session_id = session.get("session_id") or session_id_from_path(
            report_path, corpus_root
        )
        runs_by_id = {run["id"]: run for run in report.get("runs") or []}
        generation_calls = report.get("generation_calls") or []
        report_default_model = default_model_from_generation_calls(generation_calls)
        generation_calls_by_id = {call["id"]: call for call in generation_calls}

        for attempt in report.get("attempts") or []:
            pass_rate = attempt.get("test_pass_rate")
            if attempt.get("skipped") or pass_rate is None:
                continue

            run = runs_by_id.get(attempt.get("run_id"), {})
            generation_type = (
                run.get("generation_type")
                or (run.get("config") or {}).get("generation_type")
                or attempt.get("generation_type")
            )
            rows.append(
                EvidenceRow(
                    family=normalize_family(generation_type),
                    sample_id=attempt["task_id"],
                    model_name=model_name_for_attempt(
                        attempt,
                        run,
                        generation_calls_by_id,
                        report_default_model,
                    ),
                    bucket=BASELINE_BUCKET
                    if eval_run_is_baseline(run)
                    else NON_BASELINE_BUCKET,
                    pass_rate=float(pass_rate),
                    source_kind=SourceKind.EVAL_ATTEMPT,
                    session_id=session_id,
                    run_id=attempt.get("run_id") or "",
                    source_record_id=attempt.get("id") or "",
                    source_relative_path=run.get("source_relative_path") or "",
                    phase="",
                    split="",
                    generation_type=generation_type or "",
                    optimization_target="",
                )
            )
    return rows


def iter_optimizer_summary_evidence(corpus_root: Path) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for summary_path in sorted(
        corpus_root.glob("*/raw/logs/dspy_optimized/*_summary.json")
    ):
        payload = read_json_file(summary_path)
        session_id = session_id_from_path(summary_path, corpus_root)
        generation_type = payload.get("generation_type")
        run_id = summary_path.name.removesuffix("_summary.json")
        source_path = source_relative_path(summary_path, corpus_root)
        model_name = payload.get("model") or UNKNOWN_MODEL
        optimization_target = payload.get("optimization_target") or ""

        score_sources = (
            ("baseline_scores", BASELINE_BUCKET, "baseline"),
            ("optimized_scores", NON_BASELINE_BUCKET, "optimized"),
        )
        for score_key, bucket, phase in score_sources:
            for split, split_score in (payload.get(score_key) or {}).items():
                task_scores = (split_score or {}).get("task_scores") or {}
                for task_id, pass_rate in task_scores.items():
                    rows.append(
                        EvidenceRow(
                            family=normalize_family(generation_type),
                            sample_id=task_id,
                            model_name=model_name,
                            bucket=bucket,
                            pass_rate=float(pass_rate),
                            source_kind=SourceKind.OPTIMIZER_SUMMARY_TASK_SCORE,
                            session_id=session_id,
                            run_id=run_id,
                            source_record_id=(
                                f"{run_id}:{score_key}:{split}:{task_id}"
                            ),
                            source_relative_path=source_path,
                            phase=phase,
                            split=split,
                            generation_type=generation_type or "",
                            optimization_target=optimization_target,
                        )
                    )
    return rows


def iter_gepa_search_evidence(corpus_root: Path) -> list[EvidenceRow]:
    report_dir = corpus_root / "parsed_gepa_reports"
    rows: list[EvidenceRow] = []
    for report_path in sorted(report_dir.glob("*.gepa_report.json")):
        report = read_json_file(report_path)
        session = report.get("session") or {}
        session_id = session.get("session_id") or session_id_from_path(
            report_path, corpus_root
        )
        runs_by_id = {run["id"]: run for run in report.get("optimizer_runs") or []}

        for metric_call in report.get("metric_calls") or []:
            pass_rate = metric_call.get("pass_rate")
            if metric_call.get("phase") is not None or pass_rate is None:
                continue
            run = runs_by_id.get(metric_call.get("run_id"), {})
            source = metric_call.get("source") or {}
            generation_type = run.get("generation_type")
            rows.append(
                EvidenceRow(
                    family=normalize_family(generation_type),
                    sample_id=metric_call["task_id"],
                    model_name=run.get("model") or UNKNOWN_MODEL,
                    bucket=NON_BASELINE_BUCKET,
                    pass_rate=float(pass_rate),
                    source_kind=SourceKind.GEPA_SEARCH_METRIC_CALL,
                    session_id=session_id,
                    run_id=metric_call.get("run_id") or "",
                    source_record_id=metric_call.get("id") or "",
                    source_relative_path=source.get("source_relative_path") or "",
                    phase="candidate",
                    split=metric_call.get("split") or "",
                    generation_type=generation_type or "",
                    optimization_target=run.get("optimization_target") or "",
                )
            )
    return rows


def iter_mipro_search_evidence(corpus_root: Path) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for event_path in sorted(
        corpus_root.glob("*/raw/logs/dspy_optimized/*_events.jsonl")
    ):
        if "_gepa_" in event_path.name:
            continue

        summary_path = event_path.with_name(
            event_path.name.removesuffix("_events.jsonl") + "_summary.json"
        )
        if not summary_path.exists():
            continue

        summary = read_json_file(summary_path)
        generation_type = summary.get("generation_type")
        if "gepa" in (generation_type or ""):
            continue

        session_id = session_id_from_path(event_path, corpus_root)
        run_id = event_path.name.removesuffix("_events.jsonl")
        model_name = summary.get("model") or UNKNOWN_MODEL
        optimization_target = summary.get("optimization_target") or ""
        source_path = source_relative_path(event_path, corpus_root)
        in_search_window = False
        eval_phase: str | None = None

        for line_number, event in iter_jsonl(event_path):
            event_kind = event.get("event")
            payload = event.get("payload") or {}
            message = payload.get("message") or ""

            if event_kind == "step" and message.startswith("Starting "):
                in_search_window = True
                eval_phase = None
            elif event_kind == "step" and message.startswith("Evaluating "):
                if " baseline/" in message:
                    eval_phase = "baseline"
                elif " optimized/" in message:
                    eval_phase = "optimized"
                else:
                    eval_phase = None
            elif event_kind == "step" and message.startswith("Finished "):
                eval_phase = None
            elif (
                event_kind == "metric_call" and in_search_window and eval_phase is None
            ):
                pass_rate = payload.get("pass_rate")
                task_id = payload.get("task_id")
                if pass_rate is None or not task_id:
                    continue
                rows.append(
                    EvidenceRow(
                        family=normalize_family(generation_type),
                        sample_id=task_id,
                        model_name=model_name,
                        bucket=NON_BASELINE_BUCKET,
                        pass_rate=float(pass_rate),
                        source_kind=SourceKind.MIPRO_SEARCH_METRIC_CALL,
                        session_id=session_id,
                        run_id=run_id,
                        source_record_id=(
                            f"{run_id}:events:{line_number}:"
                            f"{payload.get('metric_call', '')}"
                        ),
                        source_relative_path=source_path,
                        phase="candidate",
                        split="",
                        generation_type=generation_type or "",
                        optimization_target=optimization_target,
                    )
                )
    return rows


def aggregate_evidence(rows: list[EvidenceRow]) -> list[dict[str, str | int]]:
    grouped: dict[tuple[str, str, str], AggregateValues] = {}
    for row in rows:
        key = (row.family, row.sample_id, row.model_name)
        if key not in grouped:
            grouped[key] = AggregateValues(baseline_values=[], non_baseline_values=[])
        if row.bucket == BASELINE_BUCKET:
            grouped[key].baseline_values.append(row.pass_rate)
        elif row.bucket == NON_BASELINE_BUCKET:
            grouped[key].non_baseline_values.append(row.pass_rate)
        else:
            raise ValueError(f"Unexpected bucket: {row.bucket}")

    aggregate_rows = [
        values.to_csv_row(family=family, sample_id=sample_id, model_name=model_name)
        for (family, sample_id, model_name), values in grouped.items()
    ]
    return sorted(
        aggregate_rows,
        key=lambda row: (
            str(row["direct or enc-dec"]),
            task_sort_key(str(row["sample id"])),
            str(row["model name"]),
        ),
    )


def summarize_sample_performance(rows: list[EvidenceRow]) -> list[dict[str, str | int]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.sample_id, []).append(row.pass_rate)

    return [
        {
            "sample id": sample_id,
            "average perf": mean_or_blank(values),
            "perf variance": variance_or_blank(values),
            "n average": len(values),
        }
        for sample_id, values in sorted(
            grouped.items(), key=lambda item: task_sort_key(item[0])
        )
    ]


def gpt5_nano_rows(rows: list[EvidenceRow]) -> list[EvidenceRow]:
    return [row for row in rows if row.model_name == GPT5_NANO_MODEL_NAME]


def sample_summary_outputs(
    rows: list[EvidenceRow],
) -> dict[str, list[dict[str, str | int]]]:
    rows_by_setting = sample_summary_rows_by_setting(rows)
    return {
        DEFAULT_ALL_SETTINGS_GPT5_NANO_FILE_NAME: rows_by_setting["all"],
        DEFAULT_DIRECT_GPT5_NANO_FILE_NAME: rows_by_setting["direct"],
        DEFAULT_ENCDEC_GPT5_NANO_FILE_NAME: rows_by_setting["enc-dec"],
        DEFAULT_BASELINE_GPT5_NANO_FILE_NAME: rows_by_setting["baseline"],
        DEFAULT_NON_BASELINE_GPT5_NANO_FILE_NAME: rows_by_setting["non-baseline"],
    }


def sample_summary_rows_by_setting(
    rows: list[EvidenceRow],
) -> dict[str, list[dict[str, str | int]]]:
    model_rows = gpt5_nano_rows(rows)
    return {
        "all": summarize_sample_performance(model_rows),
        "direct": summarize_sample_performance(
            [row for row in model_rows if row.family == "direct"]
        ),
        "enc-dec": summarize_sample_performance(
            [row for row in model_rows if row.family == "enc-dec"]
        ),
        "baseline": summarize_sample_performance(
            [row for row in model_rows if row.bucket == BASELINE_BUCKET]
        ),
        "non-baseline": summarize_sample_performance(
            [row for row in model_rows if row.bucket == NON_BASELINE_BUCKET]
        ),
    }


def sample_stat_json(row: dict[str, str | int]) -> dict[str, float | int]:
    return {
        "average_perf": float(row["average perf"]),
        "perf_variance": float(row["perf variance"]),
        "n_average": int(row["n average"]),
    }


def build_worst_selection_json(rows: list[EvidenceRow]) -> dict[str, Any]:
    rows_by_setting = sample_summary_rows_by_setting(rows)
    all_rows = sorted(
        rows_by_setting["all"],
        key=lambda row: (
            float(row["average perf"]),
            task_sort_key(str(row["sample id"])),
        ),
    )

    worst_sample_ids_by_n = {
        str(count): [str(row["sample id"]) for row in all_rows[:count]]
        for count in WORST_SELECTION_SIZES
    }
    selected_ids = set(worst_sample_ids_by_n[str(max(WORST_SELECTION_SIZES))])
    selection_rank_by_id = {
        str(row["sample id"]): rank
        for rank, row in enumerate(all_rows, start=1)
        if str(row["sample id"]) in selected_ids
    }

    stats_by_setting = {
        setting: {str(row["sample id"]): sample_stat_json(row) for row in setting_rows}
        for setting, setting_rows in rows_by_setting.items()
    }

    sample_stats_by_id = {
        sample_id: {
            setting: setting_stats[sample_id]
            for setting, setting_stats in stats_by_setting.items()
            if sample_id in setting_stats
        }
        for sample_id in sorted(selected_ids, key=task_sort_key)
    }

    return {
        "metadata": {
            "model_name": GPT5_NANO_MODEL_NAME,
            "selection_setting": "all",
            "selection_metric": "average perf",
            "selection_order": "ascending",
            "selection_sizes": list(WORST_SELECTION_SIZES),
            "stats_settings": list(rows_by_setting),
            "selected_id_count": len(selected_ids),
        },
        "worst_sample_ids_by_n": worst_sample_ids_by_n,
        "selection_rank_by_id": selection_rank_by_id,
        "sample_stats_by_id": sample_stats_by_id,
    }


def write_csv(
    path: Path, rows: list[dict[str, Any]], field_names: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def collect_evidence(corpus_root: Path) -> list[EvidenceRow]:
    rows = []
    rows.extend(iter_eval_attempt_evidence(corpus_root))
    rows.extend(iter_optimizer_summary_evidence(corpus_root))
    rows.extend(iter_gepa_search_evidence(corpus_root))
    rows.extend(iter_mipro_search_evidence(corpus_root))
    return sorted(
        rows,
        key=lambda row: (
            row.family,
            task_sort_key(row.sample_id),
            row.model_name,
            row.bucket,
            row.source_kind.value,
            row.session_id,
            row.source_record_id,
        ),
    )


def main(
    corpus_root: Annotated[
        Path,
        typer.Option(
            "--corpus-root",
            help="Canonical sessionized DSPy corpus root.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = DEFAULT_CORPUS_ROOT,
    evidence_output: Annotated[
        Path | None,
        typer.Option("--evidence-output", help="Evidence CSV output path."),
    ] = None,
    aggregate_output: Annotated[
        Path | None,
        typer.Option("--aggregate-output", help="Aggregate CSV output path."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Directory for default output files. Defaults to the corpus root.",
        ),
    ] = None,
) -> None:
    default_output_dir = output_dir or corpus_root
    evidence_path = output_path_or_default(
        evidence_output, default_output_dir, DEFAULT_EVIDENCE_FILE_NAME
    )
    aggregate_path = output_path_or_default(
        aggregate_output, default_output_dir, DEFAULT_AGGREGATE_FILE_NAME
    )

    evidence_rows = collect_evidence(corpus_root)
    write_csv(
        evidence_path,
        [row.to_csv_row() for row in evidence_rows],
        EVIDENCE_FIELD_NAMES,
    )

    aggregate_rows = aggregate_evidence(evidence_rows)
    write_csv(aggregate_path, aggregate_rows, AGGREGATE_FIELD_NAMES)

    typer.echo(f"Wrote evidence rows: {len(evidence_rows)} -> {evidence_path}")
    typer.echo(f"Wrote aggregate rows: {len(aggregate_rows)} -> {aggregate_path}")

    for file_name, rows in sample_summary_outputs(evidence_rows).items():
        output_path = default_output_dir / file_name
        write_csv(output_path, rows, SAMPLE_SUMMARY_FIELD_NAMES)
        typer.echo(f"Wrote sample summary rows: {len(rows)} -> {output_path}")

    worst_selection_path = (
        default_output_dir / DEFAULT_WORST_SELECTIONS_GPT5_NANO_FILE_NAME
    )
    write_json(worst_selection_path, build_worst_selection_json(evidence_rows))
    typer.echo(f"Wrote worst-selection JSON -> {worst_selection_path}")


if __name__ == "__main__":
    typer.run(main)
