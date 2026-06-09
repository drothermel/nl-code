from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nl_code.code_execution.models import TestCaseResult


LogFileKind = Literal["eval_run", "generation_history", "unknown"]
RunFormat = Literal["package", "legacy_notebook"]


class ParsedHumanEvalDspyLogFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    kind: LogFileKind
    parse_error: str | None = None
    record_count: int = 0


class HumanEvalDspyGenerationAttemptRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_type: str | None = None
    dataset_index: int | None = None
    task_id: str | None = None
    repeat_index: int | None = None
    call_index: int | None = None


class HumanEvalDspyGenerationCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file: Path
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
    usage: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None
    attempt: HumanEvalDspyGenerationAttemptRef | None = None

    @model_validator(mode="before")
    @classmethod
    def parse_raw_record(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "raw_record" not in value:
            return value

        record = value["raw_record"]
        messages = record.get("messages") or []
        usage = record.get("usage") or {}
        return {
            "source_file": value["source_file"],
            "record_index": value["record_index"],
            "timestamp": record.get("timestamp"),
            "uuid": record.get("uuid"),
            "model": record.get("model"),
            "response_model": record.get("response_model")
            or (record.get("response") or {}).get("model"),
            "model_type": record.get("model_type"),
            "prompt_fingerprint": fingerprint_messages(messages),
            "prompt_kind": prompt_kind(messages),
            "messages": messages,
            "outputs": record.get("outputs") or response_outputs(record),
            "usage": usage,
            "cost": record.get("cost") or usage.get("cost"),
            "attempt": record.get("attempt"),
        }

    @property
    def prompt_label(self) -> str:
        if self.prompt_kind:
            return self.prompt_kind
        if self.prompt_fingerprint:
            return self.prompt_fingerprint[:12]
        return "unknown"

    def row(self) -> dict[str, Any]:
        return {
            "generation_log_file": str(self.source_file),
            "generation_log_name": self.source_file.name,
            "record_index": self.record_index,
            "timestamp": self.timestamp,
            "uuid": self.uuid,
            "model": self.model,
            "response_model": self.response_model,
            "model_type": self.model_type,
            "prompt_fingerprint": self.prompt_fingerprint,
            "prompt_kind": self.prompt_kind,
            "prompt_tokens": token_count(self.usage, "prompt_tokens", "input_tokens"),
            "completion_tokens": token_count(
                self.usage,
                "completion_tokens",
                "output_tokens",
            ),
            "total_tokens": token_count(self.usage, "total_tokens"),
            "cost": self.cost,
            "attempt_generation_type": (
                self.attempt.generation_type if self.attempt else None
            ),
            "attempt_dataset_index": (
                self.attempt.dataset_index if self.attempt else None
            ),
            "attempt_task_id": self.attempt.task_id if self.attempt else None,
            "attempt_repeat_index": (
                self.attempt.repeat_index if self.attempt else None
            ),
            "attempt_call_index": self.attempt.call_index if self.attempt else None,
        }


class HumanEvalDspyAttemptSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file: Path
    run_id: str
    run_format: RunFormat
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
    test_case_results: list[TestCaseResult] = Field(default_factory=list)
    test_pass_rate: float = 0.0
    generation_log_file: Path | None = None

    @property
    def passed(self) -> bool:
        return not self.skipped and self.test_pass_rate == 1.0

    @property
    def failed_test_count(self) -> int:
        return sum(not result.passed for result in self.test_case_results)

    @property
    def first_failed_result(self) -> TestCaseResult | None:
        for result in self.test_case_results:
            if not result.passed:
                return result
        return None

    def row(self) -> dict[str, Any]:
        first_failed = self.first_failed_result
        return {
            "run_id": self.run_id,
            "run_log_file": str(self.source_file),
            "run_log_name": self.source_file.name,
            "run_format": self.run_format,
            "timestamp": self.timestamp,
            "generation_type": self.generation_type,
            "dataset_index": self.dataset_index,
            "task_id": self.task_id,
            "repeat_index": self.repeat_index,
            "skipped": self.skipped,
            "passed": self.passed,
            "pass_rate": self.test_pass_rate,
            "error": self.error,
            "test_count": len(self.test_case_results),
            "failed_test_count": self.failed_test_count,
            "first_failed_input": json_cell_value(
                first_failed.input_value if first_failed else None
            ),
            "first_failed_expected": json_cell_value(
                first_failed.expected_output if first_failed else None
            ),
            "first_failed_actual": json_cell_value(
                first_failed.actual_output if first_failed else None
            ),
            "first_failed_error": (
                first_failed.error or first_failed.compile_error
                if first_failed
                else None
            ),
            "generation_log_file": (
                str(self.generation_log_file) if self.generation_log_file else None
            ),
        }


class HumanEvalDspyRunStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_attempts: int
    evaluated_attempts: int
    skipped_count: int
    attempt_pass_count: int
    attempt_pass_rate: float
    sample_best_pass_count: int
    sample_best_pass_rate: float
    average_test_pass_rate: float


class HumanEvalDspyRunSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source_file: Path
    run_format: RunFormat
    timestamp: datetime | None = None
    generation_type: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    selected_dataset_indices: list[int] = Field(default_factory=list)
    stats_by_generation_type: dict[str, HumanEvalDspyRunStats] = Field(
        default_factory=dict
    )
    attempts: list[HumanEvalDspyAttemptSnapshot] = Field(default_factory=list)
    generation_log_files: list[Path] = Field(default_factory=list)
    prompt_fingerprints: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(default_factory=list)
    generation_call_count: int = 0

    @property
    def generation_types(self) -> list[str]:
        values = {
            attempt.generation_type
            for attempt in self.attempts
            if attempt.generation_type is not None
        }
        return sorted(values)

    def row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_log_file": str(self.source_file),
            "run_log_name": self.source_file.name,
            "run_format": self.run_format,
            "timestamp": self.timestamp,
            "generation_type": self.generation_type,
            "generation_types": ", ".join(self.generation_types),
            "model_names": ", ".join(self.model_names),
            "prompt_fingerprints": ", ".join(
                fingerprint[:12] for fingerprint in self.prompt_fingerprints
            ),
            "selected_sample_count": len(self.selected_dataset_indices),
            "attempt_count": len(self.attempts),
            "generation_call_count": self.generation_call_count,
        }


class HumanEvalDspyPipelineSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_id: str
    generation_type: str | None = None
    model: str | None = None
    prompt_fingerprints: list[str] = Field(default_factory=list)
    runs: list[HumanEvalDspyRunSnapshot] = Field(default_factory=list)

    def row(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "generation_type": self.generation_type,
            "model": self.model,
            "prompt_fingerprints": ", ".join(
                fingerprint[:12] for fingerprint in self.prompt_fingerprints
            ),
            "run_count": len(self.runs),
            "attempt_count": sum(len(run.attempts) for run in self.runs),
        }


class HumanEvalDspyLogSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    source_dir: Path
    log_files: list[ParsedHumanEvalDspyLogFile] = Field(default_factory=list)
    pipelines: list[HumanEvalDspyPipelineSnapshot] = Field(default_factory=list)
    generation_calls: list[HumanEvalDspyGenerationCall] = Field(default_factory=list)

    @property
    def runs(self) -> list[HumanEvalDspyRunSnapshot]:
        return [run for pipeline in self.pipelines for run in pipeline.runs]

    @property
    def attempts(self) -> list[HumanEvalDspyAttemptSnapshot]:
        return [attempt for run in self.runs for attempt in run.attempts]

    @property
    def failed_attempts(self) -> list[HumanEvalDspyAttemptSnapshot]:
        return [
            attempt
            for attempt in self.attempts
            if not attempt.skipped and attempt.test_pass_rate < 1.0
        ]

    @property
    def failed_task_ids(self) -> list[str]:
        return sorted(
            {
                attempt.task_id
                for attempt in self.failed_attempts
                if attempt.task_id is not None
            },
            key=human_eval_task_sort_key,
        )

    def attempts_for_task(self, task_id: str) -> list[HumanEvalDspyAttemptSnapshot]:
        return [attempt for attempt in self.attempts if attempt.task_id == task_id]

    def run_rows(self) -> list[dict[str, Any]]:
        return [run.row() for run in self.runs]

    def attempt_rows(self) -> list[dict[str, Any]]:
        return [attempt.row() for attempt in self.attempts]

    def generation_call_rows(self) -> list[dict[str, Any]]:
        return [call.row() for call in self.generation_calls]

    def pipeline_rows(self) -> list[dict[str, Any]]:
        return [pipeline.row() for pipeline in self.pipelines]

    def generation_calls_for_attempt(
        self,
        attempt: HumanEvalDspyAttemptSnapshot | None,
    ) -> list[HumanEvalDspyGenerationCall]:
        if attempt is None:
            return []
        return sorted(
            [
                call
                for call in self.generation_calls
                if call_matches_attempt(call, attempt)
            ],
            key=lambda call: (
                str(call.source_file),
                call.record_index,
            ),
        )


def parse_humaneval_dspy_logs(logs_dir: Path) -> HumanEvalDspyLogSnapshot:
    logs_dir = logs_dir.resolve()
    log_files: list[ParsedHumanEvalDspyLogFile] = []
    runs: list[HumanEvalDspyRunSnapshot] = []

    generation_calls_by_file = parse_generation_history_files(logs_dir, log_files)
    for path in iter_eval_run_paths(logs_dir):
        run = parse_eval_run_file(path, generation_calls_by_file, log_files)
        if run is not None:
            runs.append(run)

    generation_calls = [
        call for calls in generation_calls_by_file.values() for call in calls
    ]
    return HumanEvalDspyLogSnapshot(
        created_at=datetime.now(timezone.utc),
        source_dir=logs_dir,
        log_files=sorted(log_files, key=lambda item: item.path.name),
        pipelines=build_pipelines(runs),
        generation_calls=generation_calls,
    )


def write_humaneval_dspy_log_snapshot(
    snapshot: HumanEvalDspyLogSnapshot,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        snapshot.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_humaneval_dspy_log_snapshot(path: Path) -> HumanEvalDspyLogSnapshot:
    return HumanEvalDspyLogSnapshot.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def parse_generation_history_files(
    logs_dir: Path,
    log_files: list[ParsedHumanEvalDspyLogFile],
) -> dict[Path, list[HumanEvalDspyGenerationCall]]:
    calls_by_file: dict[Path, list[HumanEvalDspyGenerationCall]] = {}
    for path in sorted(logs_dir.glob("human_eval_dspy*.jsonl")):
        if path.name.endswith("_events.jsonl"):
            continue
        calls: list[HumanEvalDspyGenerationCall] = []
        parse_error = None
        for record_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_error = str(exc)
                continue
            calls.append(
                HumanEvalDspyGenerationCall.model_validate(
                    {
                        "source_file": path,
                        "record_index": record_index,
                        "raw_record": raw_record,
                    }
                )
            )
        calls_by_file[path.resolve()] = calls
        log_files.append(
            ParsedHumanEvalDspyLogFile(
                path=path,
                kind="generation_history",
                parse_error=parse_error,
                record_count=len(calls),
            )
        )
    return calls_by_file


def iter_eval_run_paths(logs_dir: Path) -> Iterable[Path]:
    yield from sorted(logs_dir.glob("human_eval_dspy_*_eval_*.json"))
    yield from sorted(logs_dir.glob("human_eval_dspy_run_*.json"))


def parse_eval_run_file(
    path: Path,
    generation_calls_by_file: dict[Path, list[HumanEvalDspyGenerationCall]],
    log_files: list[ParsedHumanEvalDspyLogFile],
) -> HumanEvalDspyRunSnapshot | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log_files.append(
            ParsedHumanEvalDspyLogFile(
                path=path,
                kind="eval_run",
                parse_error=str(exc),
            )
        )
        return None

    run = (
        parse_package_eval_run(path, payload, generation_calls_by_file)
        if "attempts" in payload
        else parse_legacy_eval_run(path, payload, generation_calls_by_file)
    )
    log_files.append(
        ParsedHumanEvalDspyLogFile(
            path=path,
            kind="eval_run",
            record_count=len(run.attempts),
        )
    )
    return run


def parse_package_eval_run(
    path: Path,
    payload: dict[str, Any],
    generation_calls_by_file: dict[Path, list[HumanEvalDspyGenerationCall]],
) -> HumanEvalDspyRunSnapshot:
    config = payload.get("config") or {}
    timestamp = payload.get("timestamp")
    run_id = run_id_for_path(path)
    attempts = [
        package_attempt(path, run_id, timestamp, attempt)
        for attempt in payload.get("attempts", [])
    ]
    run = HumanEvalDspyRunSnapshot(
        run_id=run_id,
        source_file=path,
        run_format="package",
        timestamp=timestamp,
        generation_type=config.get("generation_type"),
        config=config,
        selected_dataset_indices=payload.get("selected_dataset_indices") or [],
        stats_by_generation_type={
            name: HumanEvalDspyRunStats.model_validate(summary)
            for name, summary in (payload.get("summaries") or {}).items()
        },
        attempts=attempts,
    )
    return run_with_generation_metadata(run, generation_calls_by_file)


def parse_legacy_eval_run(
    path: Path,
    payload: dict[str, Any],
    generation_calls_by_file: dict[Path, list[HumanEvalDspyGenerationCall]],
) -> HumanEvalDspyRunSnapshot:
    timestamp = payload.get("timestamp")
    generation_type = payload.get("eval_type")
    run_id = run_id_for_path(path)
    attempts = [
        legacy_attempt(path, run_id, timestamp, generation_type, item)
        for item in payload.get("outputs", [])
    ]
    run = HumanEvalDspyRunSnapshot(
        run_id=run_id,
        source_file=path,
        run_format="legacy_notebook",
        timestamp=timestamp,
        generation_type=generation_type,
        config={
            "generation_type": generation_type,
            "n_samples": payload.get("val_num"),
            "seed": payload.get("seed"),
        },
        selected_dataset_indices=payload.get("dataset_indices") or [],
        stats_by_generation_type={str(generation_type): stats_for_attempts(attempts)},
        attempts=attempts,
    )
    return run_with_generation_metadata(run, generation_calls_by_file)


def package_attempt(
    path: Path,
    run_id: str,
    timestamp: Any,
    attempt: dict[str, Any],
) -> HumanEvalDspyAttemptSnapshot:
    return HumanEvalDspyAttemptSnapshot(
        source_file=path,
        run_id=run_id,
        run_format="package",
        timestamp=timestamp,
        generation_type=attempt.get("generation_type"),
        dataset_index=attempt.get("dataset_index"),
        task_id=attempt.get("task_id"),
        repeat_index=attempt.get("repeat_index"),
        skipped=bool(attempt.get("skipped", False)),
        error=attempt.get("error"),
        code_spec=attempt.get("code_spec"),
        raw_completed_code=attempt.get("raw_completed_code") or "",
        extracted_code=attempt.get("extracted_code") or "",
        test_case_results=[
            TestCaseResult.model_validate(result)
            for result in attempt.get("test_case_results", [])
        ],
        test_pass_rate=float(attempt.get("test_pass_rate") or 0.0),
        generation_log_file=attempt.get("generation_log_file"),
    )


def legacy_attempt(
    path: Path,
    run_id: str,
    timestamp: Any,
    generation_type: str | None,
    item: dict[str, Any],
) -> HumanEvalDspyAttemptSnapshot:
    output = item.get("output") or {}
    prediction = output.get("prediction") or {}
    return HumanEvalDspyAttemptSnapshot(
        source_file=path,
        run_id=run_id,
        run_format="legacy_notebook",
        timestamp=timestamp,
        generation_type=generation_type,
        dataset_index=item.get("dataset_index"),
        task_id=output.get("task_id"),
        repeat_index=0,
        skipped=bool(output.get("skipped", False)),
        error=output.get("error"),
        code_spec=prediction.get("code_spec"),
        raw_completed_code=prediction.get("completed_code") or "",
        extracted_code=output.get("extracted") or "",
        test_case_results=[
            TestCaseResult.model_validate(result)
            for result in output.get("results", [])
        ],
        test_pass_rate=float(output.get("pass_rate") or 0.0),
        generation_log_file=output.get("log_file"),
    )


def build_pipelines(
    runs: list[HumanEvalDspyRunSnapshot],
) -> list[HumanEvalDspyPipelineSnapshot]:
    grouped: dict[
        tuple[str | None, str | None, tuple[str, ...]],
        list[HumanEvalDspyRunSnapshot],
    ]
    grouped = defaultdict(list)
    for run in runs:
        generation_type = run.generation_type
        if generation_type is None and len(run.generation_types) == 1:
            generation_type = run.generation_types[0]
        model = run.model_names[0] if len(run.model_names) == 1 else None
        prompt_fingerprints = tuple(run.prompt_fingerprints)
        grouped[(generation_type, model, prompt_fingerprints)].append(run)

    pipelines = []
    for key, pipeline_runs in grouped.items():
        generation_type, model, prompt_fingerprints = key
        pipelines.append(
            HumanEvalDspyPipelineSnapshot(
                pipeline_id=pipeline_id_for_key(key),
                generation_type=generation_type,
                model=model,
                prompt_fingerprints=list(prompt_fingerprints),
                runs=sorted(pipeline_runs, key=lambda run: run.source_file.name),
            )
        )
    return sorted(pipelines, key=lambda pipeline: pipeline.pipeline_id)


def run_with_generation_metadata(
    run: HumanEvalDspyRunSnapshot,
    generation_calls_by_file: dict[Path, list[HumanEvalDspyGenerationCall]],
) -> HumanEvalDspyRunSnapshot:
    calls = generation_calls_for_attempts(run.attempts, generation_calls_by_file)
    relevant_calls = relevant_generation_calls(calls, run.generation_type)
    return run.model_copy(
        update={
            "generation_log_files": sorted(
                {
                    attempt.generation_log_file
                    for attempt in run.attempts
                    if attempt.generation_log_file is not None
                }
            ),
            "prompt_fingerprints": dominant_prompt_fingerprints(relevant_calls),
            "model_names": sorted(
                {call.model for call in relevant_calls if call.model}
            ),
            "generation_call_count": len(relevant_calls),
        }
    )


def relevant_generation_calls(
    calls: list[HumanEvalDspyGenerationCall],
    generation_type: str | None,
) -> list[HumanEvalDspyGenerationCall]:
    if generation_type == "direct":
        prompt_kinds = {"direct_code_from_stub"}
    elif generation_type == "encdec":
        prompt_kinds = {"encode_code_spec", "decode_code_spec"}
    else:
        prompt_kinds = set()
    if not prompt_kinds:
        return calls
    return [call for call in calls if call.prompt_kind in prompt_kinds]


def dominant_prompt_fingerprints(
    calls: list[HumanEvalDspyGenerationCall],
) -> list[str]:
    counts_by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    untyped_counts: dict[str, int] = defaultdict(int)
    for call in calls:
        if call.prompt_fingerprint is None:
            continue
        if call.prompt_kind is None:
            untyped_counts[call.prompt_fingerprint] += 1
            continue
        counts_by_kind[call.prompt_kind][call.prompt_fingerprint] += 1

    fingerprints = set()
    for counts in counts_by_kind.values():
        if counts:
            fingerprints.add(max(counts, key=lambda fingerprint: counts[fingerprint]))
    if not fingerprints and untyped_counts:
        fingerprints.add(
            max(untyped_counts, key=lambda fingerprint: untyped_counts[fingerprint])
        )
    return sorted(fingerprints)


def generation_calls_for_attempts(
    attempts: list[HumanEvalDspyAttemptSnapshot],
    generation_calls_by_file: dict[Path, list[HumanEvalDspyGenerationCall]],
) -> list[HumanEvalDspyGenerationCall]:
    files = {
        attempt.generation_log_file.resolve()
        for attempt in attempts
        if attempt.generation_log_file is not None
    }
    return [
        call
        for path in sorted(files)
        for call in generation_calls_by_file.get(path, [])
    ]


def stats_for_attempts(
    attempts: list[HumanEvalDspyAttemptSnapshot],
) -> HumanEvalDspyRunStats:
    evaluated = [attempt for attempt in attempts if not attempt.skipped]
    pass_count = sum(attempt.passed for attempt in evaluated)
    best_by_sample: dict[str, bool] = {}
    for attempt in evaluated:
        sample_id = attempt.task_id or str(attempt.dataset_index)
        best_by_sample[sample_id] = (
            best_by_sample.get(sample_id, False) or attempt.passed
        )
    return HumanEvalDspyRunStats(
        total_attempts=len(attempts),
        evaluated_attempts=len(evaluated),
        skipped_count=len(attempts) - len(evaluated),
        attempt_pass_count=pass_count,
        attempt_pass_rate=pass_count / len(evaluated) if evaluated else 0.0,
        sample_best_pass_count=sum(best_by_sample.values()),
        sample_best_pass_rate=(
            sum(best_by_sample.values()) / len(best_by_sample)
            if best_by_sample
            else 0.0
        ),
        average_test_pass_rate=(
            sum(attempt.test_pass_rate for attempt in evaluated) / len(evaluated)
            if evaluated
            else 0.0
        ),
    )


def call_matches_attempt(
    call: HumanEvalDspyGenerationCall,
    attempt: HumanEvalDspyAttemptSnapshot,
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


def response_outputs(record: dict[str, Any]) -> list[str]:
    choices = (record.get("response") or {}).get("choices") or []
    outputs = []
    for choice in choices:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            outputs.append(content)
    return outputs


def token_count(usage: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
    return None


def run_id_for_path(path: Path) -> str:
    return path.stem


def pipeline_id_for_key(
    key: tuple[str | None, str | None, tuple[str, ...]],
) -> str:
    generation_type, model, prompt_fingerprints = key
    raw_key = json.dumps(
        {
            "generation_type": generation_type,
            "model": model,
            "prompt_fingerprints": prompt_fingerprints,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
    label_parts = [
        generation_type or "mixed",
        (model or "unknown-model").split("/")[-1],
        digest,
    ]
    return "__".join(label_parts)


def human_eval_task_sort_key(task_id: str) -> tuple[str, int | str]:
    prefix, _, suffix = task_id.partition("/")
    return prefix, int(suffix) if suffix.isdigit() else suffix


def json_cell_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
