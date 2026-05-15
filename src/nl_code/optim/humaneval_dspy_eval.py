from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nl_code.code_analysis import extract_from_code_fences
from nl_code.code_execution.models import (
    CodeExecutionInfrastructureError,
    TestCase,
    TestCaseResult,
)
from nl_code.code_execution.runner import run_test_cases
from nl_code.datasets import HumanEvalDataset
from nl_code.optim.dspy_generators import (
    DEFAULT_DSPY_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_REASONING_EFFORT,
    DirectCodeGenerator,
    EncoderDecoderCodeGenerator,
    configure_dspy_lm,
)

RUN_SINGLE_TEST_CASE_FUNCTION = "run_single_test_case"


class GenerationType(StrEnum):
    DIRECT = "direct"
    ENCDEC = "encdec"
    BOTH = "both"


class HumanEvalDspyEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_type: GenerationType = GenerationType.DIRECT
    n_samples: int = 1
    seed: int = 42
    sample_indices: list[int] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    num_repeats: int = 1
    model: str = DEFAULT_DSPY_MODEL
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT
    api_base: str = DEFAULT_OPENROUTER_BASE_URL
    output_dir: Path = Path("logs")
    generation_log_file: Path | None = None
    run_log_file: Path | None = None
    timeout_seconds: float = 30.0
    docker_image: str | None = None
    log_every: int = 0

    @model_validator(mode="after")
    def validate_eval_config(self) -> HumanEvalDspyEvalConfig:
        if self.n_samples < 1:
            raise ValueError("n_samples must be positive")
        if self.num_repeats < 1:
            raise ValueError("num_repeats must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")
        if self.sample_indices and self.task_ids:
            raise ValueError("sample_indices and task_ids are mutually exclusive")
        if (self.sample_indices or self.task_ids) and self.n_samples != 1:
            raise ValueError(
                "n_samples cannot be combined with explicit sample_indices or task_ids"
            )
        return self


class HumanEvalDspyAttemptResult(BaseModel):
    generation_type: GenerationType
    dataset_index: int
    task_id: str
    repeat_index: int
    skipped: bool = False
    error: str | None = None
    code_spec: str | None = None
    raw_completed_code: str = ""
    extracted_code: str = ""
    test_case_results: list[TestCaseResult] = Field(default_factory=list)
    test_pass_rate: float = 0.0
    generation_log_file: Path | None = None


class HumanEvalDspyEvalSummary(BaseModel):
    total_attempts: int
    evaluated_attempts: int
    skipped_count: int
    attempt_pass_count: int
    attempt_pass_rate: float
    sample_best_pass_count: int
    sample_best_pass_rate: float
    average_test_pass_rate: float


class HumanEvalDspyEvalRun(BaseModel):
    timestamp: datetime
    config: HumanEvalDspyEvalConfig
    selected_dataset_indices: list[int]
    attempts: list[HumanEvalDspyAttemptResult]
    summaries: dict[str, HumanEvalDspyEvalSummary]
    run_log_file: Path | None = None


GeneratorCallable = Callable[..., Any]


def selected_generation_types(generation_type: GenerationType) -> list[GenerationType]:
    if generation_type == GenerationType.BOTH:
        return [GenerationType.DIRECT, GenerationType.ENCDEC]
    return [generation_type]


def select_dataset_indices(
    dataset: Any,
    *,
    n_samples: int,
    seed: int,
    sample_indices: Sequence[int] | None = None,
    task_ids: Sequence[str] | None = None,
) -> list[int]:
    if sample_indices and task_ids:
        raise ValueError("sample_indices and task_ids are mutually exclusive")
    if sample_indices:
        return [_normalize_dataset_index(dataset, index) for index in sample_indices]
    if task_ids:
        return [_dataset_index_for_task_id(dataset, task_id) for task_id in task_ids]

    evaluable_indices = [
        index
        for index in range(len(dataset.raw_samples))
        if _sample_at_index(dataset, index).test_results is not None
    ]
    sample_n = min(n_samples, len(evaluable_indices))
    return random.Random(seed).sample(evaluable_indices, k=sample_n)


def build_single_test_case_solution(code: str, entry_point: str) -> str:
    return "\n".join(
        [
            code.rstrip(),
            "",
            "",
            f"def {RUN_SINGLE_TEST_CASE_FUNCTION}(input_value):",
            f"    return {entry_point}(*input_value)",
            "",
        ]
    )


def build_test_cases(sample: Any) -> list[TestCase]:
    if sample.test_results is None:
        raise ValueError("sample does not provide expected test results")
    return [
        TestCase(input_value=input_value, expected_output=expected_output)
        for input_value, expected_output in zip(
            sample.test_inputs,
            sample.test_results,
            strict=True,
        )
    ]


def evaluate_completed_code(
    *,
    completed_code: str,
    sample: Any,
    timeout_seconds: float = 30.0,
    docker_image: str | None = None,
) -> tuple[str, list[TestCaseResult], float]:
    extracted_code, _had_fences = extract_from_code_fences(completed_code)
    eval_code = build_single_test_case_solution(extracted_code, sample.entry_point)
    results, pass_rate = run_test_cases(
        code=eval_code,
        function_name=RUN_SINGLE_TEST_CASE_FUNCTION,
        test_cases=build_test_cases(sample),
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
    )
    return extracted_code.rstrip() + "\n", results, pass_rate


def run_humaneval_dspy_eval(
    config: HumanEvalDspyEvalConfig,
    *,
    api_key: str | None = None,
    dataset: Any | None = None,
    direct_generator: GeneratorCallable | None = None,
    encoder_decoder_generator: GeneratorCallable | None = None,
    lm: Any | None = None,
) -> HumanEvalDspyEvalRun:
    dataset = dataset or HumanEvalDataset().load()
    generation_types = selected_generation_types(config.generation_type)
    needs_direct_generator = GenerationType.DIRECT in generation_types
    needs_encoder_decoder_generator = GenerationType.ENCDEC in generation_types
    needs_lm = (needs_direct_generator and direct_generator is None) or (
        needs_encoder_decoder_generator and encoder_decoder_generator is None
    )
    if needs_lm:
        if api_key is None:
            raise ValueError("api_key is required when generators are not provided")
        lm = configure_dspy_lm(
            model=config.model,
            api_key=api_key,
            api_base=config.api_base,
            reasoning_effort=config.reasoning_effort,
        )
    if needs_direct_generator and direct_generator is None:
        direct_generator = DirectCodeGenerator()
    if needs_encoder_decoder_generator and encoder_decoder_generator is None:
        encoder_decoder_generator = EncoderDecoderCodeGenerator()

    generation_log_file = config.generation_log_file or _timestamped_log_path(
        config.output_dir,
        "generations",
        suffix=".jsonl",
    )
    selected_indices = select_dataset_indices(
        dataset,
        n_samples=config.n_samples,
        seed=config.seed,
        sample_indices=config.sample_indices,
        task_ids=config.task_ids,
    )
    attempt_specs = [
        (generation_type, dataset_index, repeat_index)
        for generation_type in generation_types
        for dataset_index in selected_indices
        for repeat_index in range(config.num_repeats)
    ]
    attempts = []
    total_attempts = len(attempt_specs)
    for attempt_number, (
        generation_type,
        dataset_index,
        repeat_index,
    ) in enumerate(attempt_specs, start=1):
        attempts.append(
            _run_attempt(
                config=config,
                dataset=dataset,
                dataset_index=dataset_index,
                generation_type=generation_type,
                repeat_index=repeat_index,
                direct_generator=direct_generator or _unavailable_generator,
                encoder_decoder_generator=(
                    encoder_decoder_generator or _unavailable_generator
                ),
                lm=lm,
                generation_log_file=generation_log_file,
            )
        )
        log_eval_progress(
            completed_attempts=attempt_number,
            total_attempts=total_attempts,
            log_every=config.log_every,
        )
    run = HumanEvalDspyEvalRun(
        timestamp=datetime.now(timezone.utc),
        config=config,
        selected_dataset_indices=selected_indices,
        attempts=attempts,
        summaries=summarize_attempts_by_generation_type(attempts),
    )
    run_log_file = write_eval_run_log(
        run,
        config.run_log_file
        or _timestamped_log_path(config.output_dir, "run", suffix=".json"),
    )
    return run.model_copy(update={"run_log_file": run_log_file})


def summarize_attempts_by_generation_type(
    attempts: Sequence[HumanEvalDspyAttemptResult],
) -> dict[str, HumanEvalDspyEvalSummary]:
    summaries: dict[str, HumanEvalDspyEvalSummary] = {}
    for generation_type in sorted({attempt.generation_type for attempt in attempts}):
        generation_attempts = [
            attempt
            for attempt in attempts
            if attempt.generation_type == generation_type
        ]
        summaries[generation_type.value] = summarize_attempts(generation_attempts)
    return summaries


def summarize_attempts(
    attempts: Sequence[HumanEvalDspyAttemptResult],
) -> HumanEvalDspyEvalSummary:
    evaluated_attempts = [attempt for attempt in attempts if not attempt.skipped]
    attempt_pass_count = sum(
        attempt.test_pass_rate == 1.0 for attempt in evaluated_attempts
    )
    best_attempts_by_sample: dict[int, bool] = {}
    for attempt in evaluated_attempts:
        passed = attempt.test_pass_rate == 1.0
        best_attempts_by_sample[attempt.dataset_index] = (
            best_attempts_by_sample.get(attempt.dataset_index, False) or passed
        )

    evaluated_count = len(evaluated_attempts)
    sample_count = len(best_attempts_by_sample)
    return HumanEvalDspyEvalSummary(
        total_attempts=len(attempts),
        evaluated_attempts=evaluated_count,
        skipped_count=len(attempts) - evaluated_count,
        attempt_pass_count=attempt_pass_count,
        attempt_pass_rate=attempt_pass_count / evaluated_count
        if evaluated_count
        else 0.0,
        sample_best_pass_count=sum(best_attempts_by_sample.values()),
        sample_best_pass_rate=sum(best_attempts_by_sample.values()) / sample_count
        if sample_count
        else 0.0,
        average_test_pass_rate=(
            sum(attempt.test_pass_rate for attempt in evaluated_attempts)
            / evaluated_count
            if evaluated_count
            else 0.0
        ),
    )


def write_eval_run_log(run: HumanEvalDspyEvalRun, log_file: Path) -> Path:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return log_file


def log_eval_progress(
    *,
    completed_attempts: int,
    total_attempts: int,
    log_every: int,
) -> None:
    if log_every == 0:
        return
    if completed_attempts != total_attempts and completed_attempts % log_every != 0:
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    percent_complete = completed_attempts / total_attempts if total_attempts else 1.0
    print(
        f"{timestamp} completed {completed_attempts}/{total_attempts} "
        f"({percent_complete:.1%})",
        flush=True,
    )


def dump_latest_lm_history(lm_instance: Any, log_file: Path | None) -> Path | None:
    if log_file is None:
        return None
    history = getattr(lm_instance, "history", None)
    if not history:
        return None
    return dump_lm_history_since(
        lm_instance,
        log_file,
        start_index=len(history) - 1,
    )


def lm_history_length(lm_instance: Any) -> int:
    history = getattr(lm_instance, "history", None)
    return len(history) if history else 0


def dump_lm_history_since(
    lm_instance: Any,
    log_file: Path | None,
    *,
    start_index: int,
    attempt_metadata: dict[str, Any] | None = None,
) -> Path | None:
    if log_file is None:
        return None
    history = getattr(lm_instance, "history", None)
    if not history or start_index >= len(history):
        return None

    records = []
    for offset, history_record in enumerate(history[start_index:], start=start_index):
        record = dict(history_record)
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if attempt_metadata is not None:
            record["attempt"] = attempt_metadata | {"call_index": offset - start_index}
        records.append(record)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, default=_json_default, ensure_ascii=False) + "\n"
            )
    return log_file


def _run_attempt(
    *,
    config: HumanEvalDspyEvalConfig,
    dataset: Any,
    dataset_index: int,
    generation_type: GenerationType,
    repeat_index: int,
    direct_generator: GeneratorCallable,
    encoder_decoder_generator: GeneratorCallable,
    lm: Any,
    generation_log_file: Path,
) -> HumanEvalDspyAttemptResult:
    sample = _sample_at_index(dataset, dataset_index)
    if sample.test_results is None:
        return HumanEvalDspyAttemptResult(
            generation_type=generation_type,
            dataset_index=dataset_index,
            task_id=sample.task_id,
            repeat_index=repeat_index,
            skipped=True,
            error="sample does not provide expected test results",
            generation_log_file=None,
        )

    history_start_index = lm_history_length(lm)
    prediction = _generate_prediction(
        sample=sample,
        generation_type=generation_type,
        direct_generator=direct_generator,
        encoder_decoder_generator=encoder_decoder_generator,
    )
    completed_code = _prediction_field(prediction, "completed_code", default="")
    code_spec = _prediction_field(prediction, "code_spec", default=None)
    logged_history_path = dump_lm_history_since(
        lm,
        generation_log_file,
        start_index=history_start_index,
        attempt_metadata={
            "generation_type": generation_type.value,
            "dataset_index": dataset_index,
            "task_id": sample.task_id,
            "repeat_index": repeat_index,
        },
    )
    try:
        extracted_code, test_case_results, pass_rate = evaluate_completed_code(
            completed_code=completed_code,
            sample=sample,
            timeout_seconds=config.timeout_seconds,
            docker_image=config.docker_image,
        )
    except CodeExecutionInfrastructureError as exc:
        test_case_results = _failed_eval_results(build_test_cases(sample), str(exc))
        extracted_code, _had_fences = extract_from_code_fences(completed_code)
        return HumanEvalDspyAttemptResult(
            generation_type=generation_type,
            dataset_index=dataset_index,
            task_id=sample.task_id,
            repeat_index=repeat_index,
            error=str(exc),
            code_spec=code_spec,
            raw_completed_code=completed_code,
            extracted_code=extracted_code.rstrip() + "\n" if extracted_code else "",
            test_case_results=test_case_results,
            test_pass_rate=0.0,
            generation_log_file=logged_history_path,
        )

    return HumanEvalDspyAttemptResult(
        generation_type=generation_type,
        dataset_index=dataset_index,
        task_id=sample.task_id,
        repeat_index=repeat_index,
        error=_first_result_error(test_case_results),
        code_spec=code_spec,
        raw_completed_code=completed_code,
        extracted_code=extracted_code,
        test_case_results=test_case_results,
        test_pass_rate=pass_rate,
        generation_log_file=logged_history_path,
    )


def _generate_prediction(
    *,
    sample: Any,
    generation_type: GenerationType,
    direct_generator: GeneratorCallable,
    encoder_decoder_generator: GeneratorCallable,
) -> Any:
    if generation_type == GenerationType.DIRECT:
        return direct_generator(code_stub=sample.source__prompt)
    if generation_type == GenerationType.ENCDEC:
        return encoder_decoder_generator(
            input_code=sample.gt_solution,
            function_stub=sample.function_stub,
        )
    raise ValueError(f"cannot generate prediction for {generation_type!r}")


def _unavailable_generator(**_kwargs: Any) -> Any:
    raise RuntimeError("generator is unavailable for the selected generation type")


def _failed_eval_results(
    test_cases: Sequence[TestCase],
    error: str,
) -> list[TestCaseResult]:
    return [
        TestCaseResult(
            input_value=test_case.input_value,
            expected_output=test_case.expected_output,
            actual_output=None,
            passed=False,
            error=error,
            compile_success=False,
            compile_error=error,
        )
        for test_case in test_cases
    ]


def _first_result_error(results: Sequence[TestCaseResult]) -> str | None:
    for result in results:
        if result.error or result.compile_error:
            return result.error or result.compile_error
    return None


def _prediction_field(prediction: Any, field: str, *, default: Any) -> Any:
    if isinstance(prediction, dict):
        return prediction.get(field, default)
    return getattr(prediction, field, default)


def _sample_at_index(dataset: Any, index: int) -> Any:
    return dataset.get_raw_sample_at_index(index)


def _normalize_dataset_index(dataset: Any, index: int) -> int:
    size = len(dataset.raw_samples)
    normalized_index = index + size if index < 0 else index
    if normalized_index < 0 or normalized_index >= size:
        raise IndexError(f"raw sample index {index} out of range for {size} samples")
    return normalized_index


def _dataset_index_for_task_id(dataset: Any, task_id: str) -> int:
    task_ids = list(dataset.raw_samples)
    try:
        return task_ids.index(task_id)
    except ValueError as exc:
        raise KeyError(task_id) from exc


def _timestamped_log_path(output_dir: Path, name: str, *, suffix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"human_eval_dspy_{name}_{timestamp}{suffix}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "toDict"):
        return value.toDict()
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)
