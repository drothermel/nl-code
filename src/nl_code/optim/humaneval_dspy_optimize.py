from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import threading
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TextIO, cast

import dspy
from dspy.evaluate import Evaluate
from pydantic import BaseModel, ConfigDict, Field

from nl_code.code_execution.models import CodeExecutionInfrastructureError
from nl_code.datasets import HumanEvalDataset
from nl_code.optim.dspy_generators import (
    CodeSpecEncoder,
    DirectCodeGenerator,
    EncoderDecoderCodeGenerator,
    configure_dspy_lm,
)
from nl_code.datasets.humaneval_task import RawHumanEvalTask
from nl_code.optim.humaneval_dspy_eval import evaluate_completed_code
from nl_code.optim.humaneval_dspy_sample import code_stub, function_stub, gt_code

logger = logging.getLogger(__name__)

AutoMode = Literal["light", "medium", "heavy"]


class EncDecOptimizationTarget(StrEnum):
    ENCODER = "encoder"
    DECODER = "decoder"
    BOTH = "both"


class SplitTaskIds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: list[str] = Field(default_factory=list)
    dev: list[str] = Field(default_factory=list)
    eval: list[str] = Field(default_factory=list)


class SplitScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_name: str
    task_count: int
    average_pass_rate: float
    full_pass_count: int
    full_pass_rate: float
    task_scores: dict[str, float]


class OptimizationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    generation_type: str
    optimization_target: str | None = None
    model: str
    llm_config_id: str | None = None
    reasoning_config: dict[str, Any] | None = None
    auto: AutoMode | None
    max_metric_calls: int | None = None
    num_threads: int | None
    seed: int
    train_task_ids: list[str]
    dev_task_ids: list[str]
    eval_task_ids: list[str]
    baseline_scores: dict[str, SplitScore]
    optimized_scores: dict[str, SplitScore]
    optimized_program_path: Path
    summary_path: Path | None = None
    run_log_path: Path | None = None
    event_log_path: Path | None = None


class OptimizationArtifactPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stem: str
    optimized_program_path: Path
    summary_path: Path
    run_log_path: Path
    event_log_path: Path


class OptimizationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    event: str
    payload: dict[str, Any] = Field(default_factory=dict)


class OptimizationRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimized_program: Any
    summary: OptimizationSummary


class TeeTextIO:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self.streams)


class OptimizationEventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._file: TextIO | None = None

    def __enter__(self) -> OptimizationEventLogger:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, event: str, **payload: Any) -> None:
        if self._file is None:
            return
        record = OptimizationEvent(
            timestamp=datetime.now(timezone.utc),
            event=event,
            payload=json_ready(payload),
        )
        line = json.dumps(record.model_dump(mode="json"))
        with self._lock:
            self._file.write(f"{line}\n")
            self._file.flush()


_event_logger_var: contextvars.ContextVar[OptimizationEventLogger | None] = (
    contextvars.ContextVar("_event_logger", default=None)
)


class HumanEvalPassRateMetric:
    def __init__(
        self,
        *,
        samples_by_task_id: dict[str, Any],
        prediction_to_completed_code: Callable[[Any, Any], str],
        timeout_seconds: float,
        docker_image: str | None,
        verbose: bool,
        label: str,
    ) -> None:
        self.samples_by_task_id = samples_by_task_id
        self.prediction_to_completed_code = prediction_to_completed_code
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self.verbose = verbose
        self.label = label
        self._lock = threading.Lock()
        self._call_count = 0

    def __call__(
        self,
        example: dspy.Example,
        prediction: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> float:
        task_id = example.task_id
        sample = self.samples_by_task_id[task_id]
        completed_code = self.prediction_to_completed_code(example, prediction)
        try:
            _code, _results, pass_rate = evaluate_completed_code(
                completed_code=completed_code,
                sample=sample,
                timeout_seconds=self.timeout_seconds,
                docker_image=self.docker_image,
            )
        except CodeExecutionInfrastructureError as exc:
            self._log_score(task_id=task_id, pass_rate=0.0, error=str(exc))
            raise

        self._log_score(task_id=task_id, pass_rate=pass_rate, error=None)
        return pass_rate

    def _log_score(
        self,
        *,
        task_id: str,
        pass_rate: float,
        error: str | None,
    ) -> None:
        with self._lock:
            self._call_count += 1
            call_count = self._call_count
        log_optimization_event(
            "metric_call",
            label=self.label,
            metric_call=call_count,
            task_id=task_id,
            pass_rate=pass_rate,
            error=error,
        )
        if not self.verbose:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        suffix = f" error={error}" if error else ""
        print(
            f"{timestamp} [{self.label}] metric_call={call_count} "
            f"task_id={task_id} pass_rate={pass_rate:.3f}{suffix}",
            flush=True,
        )


def optimize_direct_generation(
    *,
    task_ids: SplitTaskIds,
    model: str,
    api_key: str,
    api_base: str,
    reasoning_effort: str | None,
    reasoning_config: dict[str, str | bool] | None = None,
    llm_config_id: str | None = None,
    output_dir: Path,
    auto: AutoMode | None,
    num_threads: int | None,
    seed: int,
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
    artifact_stem: str | None = None,
    run_log_path: Path | None = None,
    event_log_path: Path | None = None,
) -> OptimizationRunResult:
    configure_optimization_logging(verbose)
    log_step("Loading HumanEval dataset", verbose=verbose)
    dataset = HumanEvalDataset().load()
    samples_by_task_id = samples_for_splits(dataset, task_ids)
    trainset = direct_examples(samples_by_task_id, task_ids.train)
    devset = direct_examples(samples_by_task_id, task_ids.dev)

    log_split_sizes(task_ids, verbose=verbose)
    log_step("Configuring DSPy LM", verbose=verbose)
    lm = configure_dspy_lm(
        model=model,
        api_key=api_key,
        api_base=api_base,
        reasoning_effort=reasoning_effort,
        reasoning=reasoning_config,
    )

    baseline = DirectCodeGenerator()
    metric = direct_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label="direct/mipro",
    )

    baseline_scores = evaluate_splits(
        program=baseline,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: direct_examples(samples_by_task_id, ids),
        metric=metric,
        num_threads=num_threads,
        verbose=verbose,
        label="direct baseline",
    )
    optimized_program = compile_instruction_only(
        student=baseline,
        trainset=trainset,
        valset=devset,
        metric=metric,
        lm=lm,
        output_dir=output_dir,
        auto=auto,
        num_threads=num_threads,
        seed=seed,
        verbose=verbose,
    )
    optimized_scores = evaluate_splits(
        program=optimized_program,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: direct_examples(samples_by_task_id, ids),
        metric=metric,
        num_threads=num_threads,
        verbose=verbose,
        label="direct optimized",
    )
    return write_optimization_result(
        optimized_program=optimized_program,
        generation_type="direct",
        optimization_target=None,
        model=model,
        llm_config_id=llm_config_id,
        reasoning_config=reasoning_config,
        auto=auto,
        num_threads=num_threads,
        seed=seed,
        task_ids=task_ids,
        baseline_scores=baseline_scores,
        optimized_scores=optimized_scores,
        output_dir=output_dir,
        verbose=verbose,
        artifact_stem=artifact_stem,
        run_log_path=run_log_path,
        event_log_path=event_log_path,
    )


def optimize_encoder_decoder_generation(
    *,
    task_ids: SplitTaskIds,
    target: EncDecOptimizationTarget,
    model: str,
    api_key: str,
    api_base: str,
    reasoning_effort: str | None,
    reasoning_config: dict[str, str | bool] | None = None,
    llm_config_id: str | None = None,
    output_dir: Path,
    auto: AutoMode | None,
    num_threads: int | None,
    seed: int,
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
    artifact_stem: str | None = None,
    run_log_path: Path | None = None,
    event_log_path: Path | None = None,
) -> OptimizationRunResult:
    configure_optimization_logging(verbose)
    log_step("Loading HumanEval dataset", verbose=verbose)
    dataset = HumanEvalDataset().load()
    samples_by_task_id = samples_for_splits(dataset, task_ids)

    log_split_sizes(task_ids, verbose=verbose)
    log_step("Configuring DSPy LM", verbose=verbose)
    lm = configure_dspy_lm(
        model=model,
        api_key=api_key,
        api_base=api_base,
        reasoning_effort=reasoning_effort,
        reasoning=reasoning_config,
    )

    baseline = EncoderDecoderCodeGenerator()
    optimized_program, baseline_scores, optimized_scores = _optimize_encdec_target(
        target=target,
        baseline=baseline,
        task_ids=task_ids,
        samples_by_task_id=samples_by_task_id,
        lm=lm,
        output_dir=output_dir,
        auto=auto,
        num_threads=num_threads,
        seed=seed,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
    )
    return write_optimization_result(
        optimized_program=optimized_program,
        generation_type="encdec",
        optimization_target=target.value,
        model=model,
        llm_config_id=llm_config_id,
        reasoning_config=reasoning_config,
        auto=auto,
        num_threads=num_threads,
        seed=seed,
        task_ids=task_ids,
        baseline_scores=baseline_scores,
        optimized_scores=optimized_scores,
        output_dir=output_dir,
        verbose=verbose,
        artifact_stem=artifact_stem,
        run_log_path=run_log_path,
        event_log_path=event_log_path,
    )


def parse_task_ids(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    return [
        task_id.strip()
        for value in values
        for task_id in value.split(",")
        if task_id.strip()
    ]


def validate_disjoint_splits(task_ids: SplitTaskIds) -> None:
    split_names = ("train", "dev", "eval")
    for split_name in split_names:
        split_values = getattr(task_ids, split_name)
        duplicates = {
            task_id for task_id in split_values if split_values.count(task_id) > 1
        }
        if duplicates:
            joined = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate task IDs in {split_name} split: {joined}")

    task_id_to_splits: dict[str, list[str]] = {}
    for split_name in split_names:
        for task_id in getattr(task_ids, split_name):
            task_id_to_splits.setdefault(task_id, []).append(split_name)

    overlaps = {
        task_id: splits
        for task_id, splits in task_id_to_splits.items()
        if len(splits) > 1
    }
    if overlaps:
        details = ", ".join(
            f"{task_id} in {', '.join(splits)}"
            for task_id, splits in sorted(overlaps.items())
        )
        raise ValueError(f"task IDs must be disjoint across splits: {details}")


def require_task_ids(task_ids: SplitTaskIds) -> None:
    missing = [
        split_name
        for split_name in ("train", "dev", "eval")
        if not getattr(task_ids, split_name)
    ]
    if missing:
        joined = ", ".join(f"--{split_name}-task-ids" for split_name in missing)
        raise ValueError(f"missing required split task IDs: {joined}")
    validate_disjoint_splits(task_ids)


def normalize_auto(value: str | None) -> AutoMode | None:
    if value is None:
        return None
    if value == "none":
        raise ValueError("auto=none requires manual MIPRO settings not exposed here")
    if value in ("light", "medium", "heavy"):
        return cast(AutoMode, value)
    raise ValueError("auto must be one of: light, medium, heavy")


def direct_examples(
    samples_by_task_id: dict[str, RawHumanEvalTask],
    task_ids: Sequence[str],
) -> list[dspy.Example]:
    return [
        dspy.Example(
            task_id=task_id,
            code_stub=code_stub(samples_by_task_id[task_id]),
            completed_code=gt_code(samples_by_task_id[task_id]),
        ).with_inputs("code_stub")
        for task_id in task_ids
    ]


def encoder_examples(
    samples_by_task_id: dict[str, RawHumanEvalTask],
    task_ids: Sequence[str],
) -> list[dspy.Example]:
    return [
        dspy.Example(
            task_id=task_id,
            input_code=gt_code(samples_by_task_id[task_id]),
            function_stub=function_stub(samples_by_task_id[task_id]),
            completed_code=gt_code(samples_by_task_id[task_id]),
        ).with_inputs("input_code")
        for task_id in task_ids
    ]


def decoder_examples(
    samples_by_task_id: dict[str, RawHumanEvalTask],
    code_specs_by_task_id: dict[str, str],
    task_ids: Sequence[str],
) -> list[dspy.Example]:
    return [
        dspy.Example(
            task_id=task_id,
            code_spec=code_specs_by_task_id[task_id],
            function_stub=function_stub(samples_by_task_id[task_id]),
            completed_code=gt_code(samples_by_task_id[task_id]),
        ).with_inputs("code_spec", "function_stub")
        for task_id in task_ids
    ]


def encdec_examples(
    samples_by_task_id: dict[str, RawHumanEvalTask],
    task_ids: Sequence[str],
) -> list[dspy.Example]:
    return [
        dspy.Example(
            task_id=task_id,
            input_code=gt_code(samples_by_task_id[task_id]),
            function_stub=function_stub(samples_by_task_id[task_id]),
            completed_code=gt_code(samples_by_task_id[task_id]),
        ).with_inputs("input_code", "function_stub")
        for task_id in task_ids
    ]


def direct_metric(
    *,
    samples_by_task_id: dict[str, Any],
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
    label: str,
) -> HumanEvalPassRateMetric:
    return HumanEvalPassRateMetric(
        samples_by_task_id=samples_by_task_id,
        prediction_to_completed_code=lambda _example, prediction: prediction_field(
            prediction,
            "completed_code",
        ),
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label=label,
    )


def completed_code_metric(
    *,
    samples_by_task_id: dict[str, Any],
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
    label: str,
) -> HumanEvalPassRateMetric:
    return direct_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label=label,
    )


def encoder_metric(
    *,
    decoder: Callable[..., Any],
    samples_by_task_id: dict[str, Any],
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
    label: str,
) -> HumanEvalPassRateMetric:
    def decode_prediction(example: dspy.Example, prediction: Any) -> str:
        decoded = decoder(
            code_spec=prediction_field(prediction, "code_spec"),
            function_stub=example.function_stub,
        )
        return prediction_field(decoded, "completed_code")

    return HumanEvalPassRateMetric(
        samples_by_task_id=samples_by_task_id,
        prediction_to_completed_code=decode_prediction,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label=label,
    )


def compile_instruction_only(
    *,
    student: Any,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    metric: Callable[..., float],
    lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    num_threads: int | None,
    seed: int,
    verbose: bool,
) -> Any:
    mipro_log_dir = output_dir / "mipro_logs" / timestamp_slug()
    log_optimization_event("mipro_log_dir", path=mipro_log_dir)
    log_step(
        (
            "Starting MIPROv2 instruction-only optimization "
            f"auto={auto!r} train={len(trainset)} dev={len(valset)} "
            f"num_threads={num_threads} log_dir={mipro_log_dir}"
        ),
        verbose=verbose,
    )
    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=lm,
        task_model=lm,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        auto=auto,
        num_threads=num_threads,
        seed=seed,
        verbose=verbose,
        log_dir=str(mipro_log_dir),
    )
    optimized = optimizer.compile(
        student,
        trainset=trainset,
        valset=valset,
        seed=seed,
    )
    log_step("Finished MIPROv2 optimization", verbose=verbose)
    return optimized


def evaluate_splits(
    *,
    program: Any,
    task_ids: SplitTaskIds,
    examples_for_task_ids: Callable[[Sequence[str]], list[dspy.Example]],
    metric: Callable[..., Any],
    num_threads: int | None,
    verbose: bool,
    label: str,
) -> dict[str, SplitScore]:
    scores = {}
    for split_name in ("train", "dev", "eval"):
        split_task_ids = getattr(task_ids, split_name)
        examples = examples_for_task_ids(split_task_ids)
        scores[split_name] = evaluate_examples(
            program=program,
            examples=examples,
            metric=metric,
            num_threads=num_threads,
            verbose=verbose,
            label=f"{label}/{split_name}",
        )
    return scores


def evaluate_examples(
    *,
    program: Any,
    examples: list[dspy.Example],
    metric: Callable[..., Any],
    num_threads: int | None,
    verbose: bool,
    label: str,
) -> SplitScore:
    log_step(
        f"Evaluating {label}: tasks={len(examples)} num_threads={num_threads}",
        verbose=verbose,
    )
    evaluator = Evaluate(
        devset=examples,
        metric=metric,
        num_threads=num_threads,
        display_progress=verbose,
        display_table=False,
    )
    result = evaluator(program)
    task_scores = {
        example.task_id: score_value(score)
        for example, _prediction, score in result.results
    }
    task_count = len(task_scores)
    full_pass_count = sum(score == 1.0 for score in task_scores.values())
    split_score = SplitScore(
        split_name=label,
        task_count=task_count,
        average_pass_rate=(
            sum(task_scores.values()) / task_count if task_count else 0.0
        ),
        full_pass_count=full_pass_count,
        full_pass_rate=full_pass_count / task_count if task_count else 0.0,
        task_scores=task_scores,
    )
    log_optimization_event(
        "split_score",
        label=label,
        split_score=split_score.model_dump(mode="json"),
    )
    log_step(
        (
            f"Finished {label}: average_pass_rate="
            f"{split_score.average_pass_rate:.3f} full_pass_rate="
            f"{split_score.full_pass_rate:.3f}"
        ),
        verbose=verbose,
    )
    return split_score


def samples_for_splits(dataset: Any, task_ids: SplitTaskIds) -> dict[str, Any]:
    selected_task_ids = task_ids.train + task_ids.dev + task_ids.eval
    missing = [
        task_id for task_id in selected_task_ids if task_id not in dataset.raw_samples
    ]
    if missing:
        raise KeyError(f"unknown HumanEval task IDs: {', '.join(missing)}")
    return {
        task_id: dataset.raw_samples[task_id]
        for task_id in dict.fromkeys(selected_task_ids)
    }


def precompute_code_specs(
    *,
    encoder: CodeSpecEncoder,
    samples_by_task_id: dict[str, Any],
    task_ids: Sequence[str],
    verbose: bool,
) -> dict[str, str]:
    code_specs = {}
    for index, task_id in enumerate(task_ids, start=1):
        log_step(
            f"Precomputing code spec {index}/{len(task_ids)} task_id={task_id}",
            verbose=verbose,
        )
        prediction = encoder(input_code=gt_code(samples_by_task_id[task_id]))
        code_specs[task_id] = prediction_field(prediction, "code_spec")
    return code_specs


def prediction_field(prediction: Any, field: str) -> str:
    if isinstance(prediction, dict):
        value = prediction.get(field, "")
    else:
        value = getattr(prediction, field, "")
    return str(value or "")


def score_value(score: Any) -> float:
    if isinstance(score, dict):
        return float(score.get("score", 0.0))
    if hasattr(score, "score"):
        return float(score.score)
    return float(score)


def configure_optimization_logging(verbose: bool) -> None:
    if not verbose:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def log_split_sizes(task_ids: SplitTaskIds, *, verbose: bool) -> None:
    log_step(
        (
            f"Split sizes train={len(task_ids.train)} dev={len(task_ids.dev)} "
            f"eval={len(task_ids.eval)}"
        ),
        verbose=verbose,
    )


def log_step(message: str, *, verbose: bool) -> None:
    log_optimization_event("step", message=message)
    if not verbose:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"{timestamp} {message}", flush=True)


def optimization_artifact_paths(
    *,
    output_dir: Path,
    generation_type: str,
    optimization_target: str | None,
    timestamp: str | None = None,
) -> OptimizationArtifactPaths:
    stem = optimization_artifact_stem(
        generation_type=generation_type,
        optimization_target=optimization_target,
        timestamp=timestamp,
    )
    return OptimizationArtifactPaths(
        stem=stem,
        optimized_program_path=output_dir / f"{stem}.json",
        summary_path=output_dir / f"{stem}_summary.json",
        run_log_path=output_dir / f"{stem}_run.log",
        event_log_path=output_dir / f"{stem}_events.jsonl",
    )


def optimization_artifact_stem(
    *,
    generation_type: str,
    optimization_target: str | None,
    timestamp: str | None = None,
) -> str:
    slug_parts = ["human_eval_dspy", generation_type]
    if optimization_target:
        slug_parts.append(optimization_target)
    slug_parts.append("optimized")
    slug_parts.append(timestamp or timestamp_slug())
    return "_".join(slug_parts)


@contextmanager
def optimization_log_context(
    *,
    run_log_path: Path,
    event_log_path: Path,
) -> Iterator[None]:
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        run_log_path.open("w", encoding="utf-8") as run_log_file,
        OptimizationEventLogger(event_log_path) as event_logger,
    ):
        event_logger_token = _event_logger_var.set(event_logger)
        try:
            with (
                redirect_stdout(TeeTextIO(sys.stdout, run_log_file)),
                redirect_stderr(TeeTextIO(sys.stderr, run_log_file)),
            ):
                log_optimization_event(
                    "run_start",
                    run_log_path=run_log_path,
                    event_log_path=event_log_path,
                )
                try:
                    yield
                except BaseException as exc:
                    log_optimization_event(
                        "run_error",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback="".join(
                            traceback.format_exception(
                                type(exc), exc, exc.__traceback__
                            )
                        ),
                    )
                    raise
                finally:
                    log_optimization_event("run_end")
        finally:
            _event_logger_var.reset(event_logger_token)
            logging.basicConfig(level=logging.WARNING, force=True)


def log_optimization_event(event: str, **payload: Any) -> None:
    event_logger = _event_logger_var.get()
    if event_logger is not None:
        event_logger.write(event, **payload)


def json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_optimization_result(
    *,
    optimized_program: Any,
    generation_type: str,
    optimization_target: str | None,
    model: str,
    llm_config_id: str | None = None,
    reasoning_config: dict[str, Any] | None = None,
    auto: AutoMode | None,
    max_metric_calls: int | None = None,
    num_threads: int | None,
    seed: int,
    task_ids: SplitTaskIds,
    baseline_scores: dict[str, SplitScore],
    optimized_scores: dict[str, SplitScore],
    output_dir: Path,
    verbose: bool,
    artifact_stem: str | None = None,
    run_log_path: Path | None = None,
    event_log_path: Path | None = None,
) -> OptimizationRunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem or optimization_artifact_stem(
        generation_type=generation_type,
        optimization_target=optimization_target,
    )
    program_path = output_dir / f"{stem}.json"
    summary_path = output_dir / f"{stem}_summary.json"
    optimized_program.save(program_path)
    summary = OptimizationSummary(
        timestamp=datetime.now(timezone.utc),
        generation_type=generation_type,
        optimization_target=optimization_target,
        model=model,
        llm_config_id=llm_config_id,
        reasoning_config=reasoning_config,
        auto=auto,
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
        seed=seed,
        train_task_ids=task_ids.train,
        dev_task_ids=task_ids.dev,
        eval_task_ids=task_ids.eval,
        baseline_scores=baseline_scores,
        optimized_scores=optimized_scores,
        optimized_program_path=program_path,
        summary_path=summary_path,
        run_log_path=run_log_path,
        event_log_path=event_log_path,
    )
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    log_optimization_event(
        "artifacts_saved",
        optimized_program_path=program_path,
        summary_path=summary_path,
        run_log_path=run_log_path,
        event_log_path=event_log_path,
    )
    log_step(f"Saved optimized program: {program_path}", verbose=verbose)
    log_step(f"Saved optimization summary: {summary_path}", verbose=verbose)
    return OptimizationRunResult(optimized_program=optimized_program, summary=summary)


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _optimize_encdec_target(
    *,
    target: EncDecOptimizationTarget,
    baseline: EncoderDecoderCodeGenerator,
    task_ids: SplitTaskIds,
    samples_by_task_id: dict[str, Any],
    lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    num_threads: int | None,
    seed: int,
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
) -> tuple[Any, dict[str, SplitScore], dict[str, SplitScore]]:
    if target == EncDecOptimizationTarget.ENCODER:
        return _optimize_encoder(
            baseline=baseline,
            task_ids=task_ids,
            samples_by_task_id=samples_by_task_id,
            lm=lm,
            output_dir=output_dir,
            auto=auto,
            num_threads=num_threads,
            seed=seed,
            timeout_seconds=timeout_seconds,
            docker_image=docker_image,
            verbose=verbose,
        )
    if target == EncDecOptimizationTarget.DECODER:
        return _optimize_decoder(
            baseline=baseline,
            task_ids=task_ids,
            samples_by_task_id=samples_by_task_id,
            lm=lm,
            output_dir=output_dir,
            auto=auto,
            num_threads=num_threads,
            seed=seed,
            timeout_seconds=timeout_seconds,
            docker_image=docker_image,
            verbose=verbose,
        )
    return _optimize_both(
        baseline=baseline,
        task_ids=task_ids,
        samples_by_task_id=samples_by_task_id,
        lm=lm,
        output_dir=output_dir,
        auto=auto,
        num_threads=num_threads,
        seed=seed,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
    )


def _optimize_encoder(
    *,
    baseline: EncoderDecoderCodeGenerator,
    task_ids: SplitTaskIds,
    samples_by_task_id: dict[str, Any],
    lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    num_threads: int | None,
    seed: int,
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
) -> tuple[Any, dict[str, SplitScore], dict[str, SplitScore]]:
    metric = encoder_metric(
        decoder=baseline.decoder,
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label="encdec_encoder/mipro",
    )
    baseline_scores = evaluate_splits(
        program=baseline.encoder,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: encoder_examples(samples_by_task_id, ids),
        metric=metric,
        num_threads=num_threads,
        verbose=verbose,
        label="encoder baseline",
    )
    optimized_encoder = compile_instruction_only(
        student=baseline.encoder,
        trainset=encoder_examples(samples_by_task_id, task_ids.train),
        valset=encoder_examples(samples_by_task_id, task_ids.dev),
        metric=metric,
        lm=lm,
        output_dir=output_dir,
        auto=auto,
        num_threads=num_threads,
        seed=seed,
        verbose=verbose,
    )
    optimized_scores = evaluate_splits(
        program=optimized_encoder,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: encoder_examples(samples_by_task_id, ids),
        metric=metric,
        num_threads=num_threads,
        verbose=verbose,
        label="encoder optimized",
    )
    return optimized_encoder, baseline_scores, optimized_scores


def _optimize_decoder(
    *,
    baseline: EncoderDecoderCodeGenerator,
    task_ids: SplitTaskIds,
    samples_by_task_id: dict[str, Any],
    lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    num_threads: int | None,
    seed: int,
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
) -> tuple[Any, dict[str, SplitScore], dict[str, SplitScore]]:
    all_task_ids = task_ids.train + task_ids.dev + task_ids.eval
    code_specs = precompute_code_specs(
        encoder=baseline.encoder,
        samples_by_task_id=samples_by_task_id,
        task_ids=all_task_ids,
        verbose=verbose,
    )
    metric = completed_code_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label="encdec_decoder/mipro",
    )

    def examples_for_ids(ids: Sequence[str]) -> list[dspy.Example]:
        return decoder_examples(samples_by_task_id, code_specs, ids)

    baseline_scores = evaluate_splits(
        program=baseline.decoder,
        task_ids=task_ids,
        examples_for_task_ids=examples_for_ids,
        metric=metric,
        num_threads=num_threads,
        verbose=verbose,
        label="decoder baseline",
    )
    optimized_decoder = compile_instruction_only(
        student=baseline.decoder,
        trainset=examples_for_ids(task_ids.train),
        valset=examples_for_ids(task_ids.dev),
        metric=metric,
        lm=lm,
        output_dir=output_dir,
        auto=auto,
        num_threads=num_threads,
        seed=seed,
        verbose=verbose,
    )
    optimized_scores = evaluate_splits(
        program=optimized_decoder,
        task_ids=task_ids,
        examples_for_task_ids=examples_for_ids,
        metric=metric,
        num_threads=num_threads,
        verbose=verbose,
        label="decoder optimized",
    )
    return optimized_decoder, baseline_scores, optimized_scores


def _optimize_both(
    *,
    baseline: EncoderDecoderCodeGenerator,
    task_ids: SplitTaskIds,
    samples_by_task_id: dict[str, Any],
    lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    num_threads: int | None,
    seed: int,
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
) -> tuple[Any, dict[str, SplitScore], dict[str, SplitScore]]:
    metric = completed_code_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label="encdec_both/mipro",
    )
    baseline_scores = evaluate_splits(
        program=baseline,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: encdec_examples(samples_by_task_id, ids),
        metric=metric,
        num_threads=num_threads,
        verbose=verbose,
        label="encdec baseline",
    )
    optimized_program = compile_instruction_only(
        student=baseline,
        trainset=encdec_examples(samples_by_task_id, task_ids.train),
        valset=encdec_examples(samples_by_task_id, task_ids.dev),
        metric=metric,
        lm=lm,
        output_dir=output_dir,
        auto=auto,
        num_threads=num_threads,
        seed=seed,
        verbose=verbose,
    )
    optimized_scores = evaluate_splits(
        program=optimized_program,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: encdec_examples(samples_by_task_id, ids),
        metric=metric,
        num_threads=num_threads,
        verbose=verbose,
        label="encdec optimized",
    )
    return optimized_program, baseline_scores, optimized_scores


def api_key_from_env() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY must be set")
    if api_key.strip() in {"...", "REPLACE_ME", "YOUR_KEY"}:
        raise ValueError("OPENROUTER_API_KEY is still a placeholder")
    return api_key
