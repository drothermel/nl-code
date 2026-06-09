from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from nl_code.code_execution.models import (
    CodeExecutionInfrastructureError,
    TestCaseResult,
)
from nl_code.datasets import HumanEvalDataset
from nl_code.optim.dspy_generators import (
    CodeSpecDecoder,
    DirectCodeGenerator,
    EncoderDecoderCodeGenerator,
    configure_dspy_lm,
)
from nl_code.optim.humaneval_dspy_eval import evaluate_completed_code
from nl_code.optim.humaneval_dspy_optimize import (
    AutoMode,
    EncDecOptimizationTarget,
    OptimizationRunResult,
    SplitScore,
    SplitTaskIds,
    completed_code_metric,
    configure_optimization_logging,
    decoder_examples,
    direct_examples,
    encoder_examples,
    encoder_metric,
    encdec_examples,
    evaluate_splits,
    log_optimization_event,
    log_split_sizes,
    log_step,
    precompute_code_specs,
    prediction_field,
    samples_for_splits,
    timestamp_slug,
    write_optimization_result,
)


class HumanEvalGepaFeedbackMetric:
    def __init__(
        self,
        *,
        samples_by_task_id: dict[str, Any],
        prediction_to_completed_code: Callable[[dspy.Example, Any], str],
        timeout_seconds: float,
        docker_image: str | None,
        label: str,
        verbose: bool = False,
    ) -> None:
        self.samples_by_task_id = samples_by_task_id
        self.prediction_to_completed_code = prediction_to_completed_code
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self.label = label
        self.verbose = verbose
        self._lock = threading.Lock()
        self._call_count = 0

    def __call__(
        self,
        gold: dspy.Example,
        pred: Any,
        trace: Any | None = None,
        pred_name: str | None = None,
        pred_trace: Any | None = None,
    ) -> ScoreWithFeedback:
        del trace, pred_trace
        task_id = gold.task_id
        sample = self.samples_by_task_id[task_id]
        completed_code = self.prediction_to_completed_code(gold, pred)
        try:
            extracted_code, test_results, pass_rate = evaluate_completed_code(
                completed_code=completed_code,
                sample=sample,
                timeout_seconds=self.timeout_seconds,
                docker_image=self.docker_image,
            )
        except CodeExecutionInfrastructureError as exc:
            self._log_score(
                task_id=task_id,
                pred_name=pred_name,
                pass_rate=0.0,
                error=str(exc),
            )
            raise

        self._log_score(
            task_id=task_id,
            pred_name=pred_name,
            pass_rate=pass_rate,
            error=None,
        )
        feedback = self._feedback(
            task_id=task_id,
            pred_name=pred_name,
            pass_rate=pass_rate,
            test_results=test_results,
            extracted_code=extracted_code,
        )
        return ScoreWithFeedback(score=pass_rate, feedback=feedback)

    def _feedback(
        self,
        *,
        task_id: str,
        pred_name: str | None,
        pass_rate: float,
        test_results: Sequence[TestCaseResult],
        extracted_code: str,
    ) -> str:
        prefix = self._prefix(task_id, pred_name)
        if pass_rate == 1.0:
            return (
                f"{prefix} All available tests passed. Preserve the behavior, "
                "signature, and executable-code-only output format."
            )

        first_failed = next(
            (result for result in test_results if not result.passed),
            None,
        )
        parts = [
            f"{prefix} Generated code failed tests with pass_rate={pass_rate:.3f}.",
            self._predictor_hint(pred_name),
        ]
        if first_failed is not None:
            parts.append(_failed_result_feedback(first_failed))
        if extracted_code:
            parts.append(f"Generated code excerpt:\n{_truncate(extracted_code, 1200)}")
        return "\n".join(part for part in parts if part)

    def _prefix(self, task_id: str, pred_name: str | None) -> str:
        target = f" predictor={pred_name}" if pred_name else ""
        return f"[{self.label} task_id={task_id}{target}]"

    def _predictor_hint(self, pred_name: str | None) -> str:
        if pred_name is None:
            return (
                "Revise the program instructions so the final completed_code is "
                "plain executable Python that exactly implements the task."
            )
        if "encoder" in pred_name:
            return (
                "This feedback is for the encoder. Improve the natural-language "
                "code_spec so it preserves exact behavior, edge cases, return "
                "types, and tie-breaking needed by the decoder."
            )
        if "decoder" in pred_name:
            return (
                "This feedback is for the decoder. Use the code_spec and "
                "function_stub to emit only executable Python code with the "
                "original signature and correct edge-case behavior."
            )
        return (
            "Revise this predictor's instruction so the downstream program "
            "produces correct executable Python."
        )

    def _log_score(
        self,
        *,
        task_id: str,
        pred_name: str | None,
        pass_rate: float,
        error: str | None,
    ) -> None:
        with self._lock:
            self._call_count += 1
            call_count = self._call_count
        log_optimization_event(
            "gepa_metric_call",
            label=self.label,
            metric_call=call_count,
            task_id=task_id,
            predictor=pred_name,
            pass_rate=pass_rate,
            error=error,
        )
        if not self.verbose:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        predictor = f" predictor={pred_name}" if pred_name else ""
        suffix = f" error={error}" if error else ""
        print(
            f"{timestamp} [{self.label}] metric_call={call_count} "
            f"task_id={task_id}{predictor} pass_rate={pass_rate:.3f}{suffix}",
            flush=True,
        )


def optimize_direct_generation_gepa(
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
    max_metric_calls: int | None,
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

    log_step("Configuring DSPy LM and GEPA reflection LM", verbose=verbose)
    lm = configure_dspy_lm(
        model=model,
        llm_config_id=llm_config_id,
        api_key=api_key,
        api_base=api_base,
        reasoning_effort=reasoning_effort,
        reasoning=reasoning_config,
    )
    baseline = DirectCodeGenerator()
    eval_metric = completed_code_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label="direct/gepa-eval",
    )
    gepa_metric = direct_gepa_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
    )

    baseline_scores = evaluate_splits(
        program=baseline,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: direct_examples(samples_by_task_id, ids),
        metric=eval_metric,
        num_threads=num_threads,
        verbose=verbose,
        label="direct GEPA baseline",
    )
    optimized_program = compile_gepa(
        student=baseline,
        trainset=direct_examples(samples_by_task_id, task_ids.train),
        valset=direct_examples(samples_by_task_id, task_ids.dev),
        metric=gepa_metric,
        reflection_lm=lm,
        output_dir=output_dir,
        auto=auto,
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
        seed=seed,
        verbose=verbose,
    )
    optimized_scores = evaluate_splits(
        program=optimized_program,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: direct_examples(samples_by_task_id, ids),
        metric=eval_metric,
        num_threads=num_threads,
        verbose=verbose,
        label="direct GEPA optimized",
    )
    return write_optimization_result(
        optimized_program=optimized_program,
        generation_type="direct_gepa",
        optimization_target=None,
        model=model,
        llm_config_id=llm_config_id,
        reasoning_config=reasoning_config,
        auto=auto,
        max_metric_calls=max_metric_calls,
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


def optimize_encoder_decoder_generation_gepa(
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
    max_metric_calls: int | None,
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

    log_step("Configuring DSPy LM and GEPA reflection LM", verbose=verbose)
    lm = configure_dspy_lm(
        model=model,
        llm_config_id=llm_config_id,
        api_key=api_key,
        api_base=api_base,
        reasoning_effort=reasoning_effort,
        reasoning=reasoning_config,
    )
    baseline = EncoderDecoderCodeGenerator()
    optimized_program, baseline_scores, optimized_scores = _optimize_encdec_gepa(
        target=target,
        baseline=baseline,
        task_ids=task_ids,
        samples_by_task_id=samples_by_task_id,
        reflection_lm=lm,
        output_dir=output_dir,
        auto=auto,
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
        seed=seed,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
    )
    return write_optimization_result(
        optimized_program=optimized_program,
        generation_type="encdec_gepa",
        optimization_target=target.value,
        model=model,
        llm_config_id=llm_config_id,
        reasoning_config=reasoning_config,
        auto=auto,
        max_metric_calls=max_metric_calls,
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


def direct_gepa_metric(
    *,
    samples_by_task_id: dict[str, Any],
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool = False,
) -> HumanEvalGepaFeedbackMetric:
    return HumanEvalGepaFeedbackMetric(
        samples_by_task_id=samples_by_task_id,
        prediction_to_completed_code=lambda _example, prediction: prediction_field(
            prediction,
            "completed_code",
        ),
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        label="direct-gepa",
        verbose=verbose,
    )


def encoder_gepa_metric(
    *,
    decoder: CodeSpecDecoder,
    samples_by_task_id: dict[str, Any],
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool = False,
) -> HumanEvalGepaFeedbackMetric:
    def decode_prediction(example: dspy.Example, prediction: Any) -> str:
        decoded = decoder(
            code_spec=prediction_field(prediction, "code_spec"),
            function_stub=example.function_stub,
        )
        return prediction_field(decoded, "completed_code")

    return HumanEvalGepaFeedbackMetric(
        samples_by_task_id=samples_by_task_id,
        prediction_to_completed_code=decode_prediction,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        label="encoder-gepa",
        verbose=verbose,
    )


def compile_gepa(
    *,
    student: Any,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    metric: HumanEvalGepaFeedbackMetric,
    reflection_lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    max_metric_calls: int | None,
    num_threads: int | None,
    seed: int,
    verbose: bool,
) -> Any:
    gepa_log_dir = output_dir / "gepa_logs" / timestamp_slug()
    log_step(
        (
            "Starting GEPA optimization "
            f"auto={auto!r} max_metric_calls={max_metric_calls} "
            f"train={len(trainset)} dev={len(valset)} "
            f"num_threads={num_threads} log_dir={gepa_log_dir}"
        ),
        verbose=verbose,
    )
    optimizer = dspy.GEPA(
        metric=metric,
        auto=auto,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        num_threads=num_threads,
        log_dir=str(gepa_log_dir),
        track_stats=True,
        seed=seed,
    )
    optimized = optimizer.compile(student, trainset=trainset, valset=valset)
    log_step("Finished GEPA optimization", verbose=verbose)
    return optimized


def _optimize_encdec_gepa(
    *,
    target: EncDecOptimizationTarget,
    baseline: EncoderDecoderCodeGenerator,
    task_ids: SplitTaskIds,
    samples_by_task_id: dict[str, Any],
    reflection_lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    max_metric_calls: int | None,
    num_threads: int | None,
    seed: int,
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
) -> tuple[Any, dict[str, SplitScore], dict[str, SplitScore]]:
    if target == EncDecOptimizationTarget.ENCODER:
        return _optimize_encoder_gepa(
            baseline=baseline,
            task_ids=task_ids,
            samples_by_task_id=samples_by_task_id,
            reflection_lm=reflection_lm,
            output_dir=output_dir,
            auto=auto,
            max_metric_calls=max_metric_calls,
            num_threads=num_threads,
            seed=seed,
            timeout_seconds=timeout_seconds,
            docker_image=docker_image,
            verbose=verbose,
        )
    if target == EncDecOptimizationTarget.DECODER:
        return _optimize_decoder_gepa(
            baseline=baseline,
            task_ids=task_ids,
            samples_by_task_id=samples_by_task_id,
            reflection_lm=reflection_lm,
            output_dir=output_dir,
            auto=auto,
            max_metric_calls=max_metric_calls,
            num_threads=num_threads,
            seed=seed,
            timeout_seconds=timeout_seconds,
            docker_image=docker_image,
            verbose=verbose,
        )
    return _optimize_both_gepa(
        baseline=baseline,
        task_ids=task_ids,
        samples_by_task_id=samples_by_task_id,
        reflection_lm=reflection_lm,
        output_dir=output_dir,
        auto=auto,
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
        seed=seed,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
    )


def _optimize_encoder_gepa(
    *,
    baseline: EncoderDecoderCodeGenerator,
    task_ids: SplitTaskIds,
    samples_by_task_id: dict[str, Any],
    reflection_lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    max_metric_calls: int | None,
    num_threads: int | None,
    seed: int,
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
) -> tuple[Any, dict[str, SplitScore], dict[str, SplitScore]]:
    eval_metric = encoder_metric(
        decoder=baseline.decoder,
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label="encoder/gepa-eval",
    )
    gepa_metric = encoder_gepa_metric(
        decoder=baseline.decoder,
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
    )
    baseline_scores = evaluate_splits(
        program=baseline.encoder,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: encoder_examples(samples_by_task_id, ids),
        metric=eval_metric,
        num_threads=num_threads,
        verbose=verbose,
        label="encoder GEPA baseline",
    )
    optimized_encoder = compile_gepa(
        student=baseline.encoder,
        trainset=encoder_examples(samples_by_task_id, task_ids.train),
        valset=encoder_examples(samples_by_task_id, task_ids.dev),
        metric=gepa_metric,
        reflection_lm=reflection_lm,
        output_dir=output_dir,
        auto=auto,
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
        seed=seed,
        verbose=verbose,
    )
    optimized_scores = evaluate_splits(
        program=optimized_encoder,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: encoder_examples(samples_by_task_id, ids),
        metric=eval_metric,
        num_threads=num_threads,
        verbose=verbose,
        label="encoder GEPA optimized",
    )
    return optimized_encoder, baseline_scores, optimized_scores


def _optimize_decoder_gepa(
    *,
    baseline: EncoderDecoderCodeGenerator,
    task_ids: SplitTaskIds,
    samples_by_task_id: dict[str, Any],
    reflection_lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    max_metric_calls: int | None,
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

    def examples_for_ids(ids: Sequence[str]) -> list[dspy.Example]:
        return decoder_examples(samples_by_task_id, code_specs, ids)

    eval_metric = completed_code_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label="encdec_decoder/gepa-eval",
    )
    gepa_metric = direct_gepa_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
    )
    baseline_scores = evaluate_splits(
        program=baseline.decoder,
        task_ids=task_ids,
        examples_for_task_ids=examples_for_ids,
        metric=eval_metric,
        num_threads=num_threads,
        verbose=verbose,
        label="decoder GEPA baseline",
    )
    optimized_decoder = compile_gepa(
        student=baseline.decoder,
        trainset=examples_for_ids(task_ids.train),
        valset=examples_for_ids(task_ids.dev),
        metric=gepa_metric,
        reflection_lm=reflection_lm,
        output_dir=output_dir,
        auto=auto,
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
        seed=seed,
        verbose=verbose,
    )
    optimized_scores = evaluate_splits(
        program=optimized_decoder,
        task_ids=task_ids,
        examples_for_task_ids=examples_for_ids,
        metric=eval_metric,
        num_threads=num_threads,
        verbose=verbose,
        label="decoder GEPA optimized",
    )
    return optimized_decoder, baseline_scores, optimized_scores


def _optimize_both_gepa(
    *,
    baseline: EncoderDecoderCodeGenerator,
    task_ids: SplitTaskIds,
    samples_by_task_id: dict[str, Any],
    reflection_lm: Any,
    output_dir: Path,
    auto: AutoMode | None,
    max_metric_calls: int | None,
    num_threads: int | None,
    seed: int,
    timeout_seconds: float,
    docker_image: str | None,
    verbose: bool,
) -> tuple[Any, dict[str, SplitScore], dict[str, SplitScore]]:
    eval_metric = completed_code_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
        label="encdec/gepa-eval",
    )
    gepa_metric = direct_gepa_metric(
        samples_by_task_id=samples_by_task_id,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
    )
    baseline_scores = evaluate_splits(
        program=baseline,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: encdec_examples(samples_by_task_id, ids),
        metric=eval_metric,
        num_threads=num_threads,
        verbose=verbose,
        label="encdec GEPA baseline",
    )
    optimized_program = compile_gepa(
        student=baseline,
        trainset=encdec_examples(samples_by_task_id, task_ids.train),
        valset=encdec_examples(samples_by_task_id, task_ids.dev),
        metric=gepa_metric,
        reflection_lm=reflection_lm,
        output_dir=output_dir,
        auto=auto,
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
        seed=seed,
        verbose=verbose,
    )
    optimized_scores = evaluate_splits(
        program=optimized_program,
        task_ids=task_ids,
        examples_for_task_ids=lambda ids: encdec_examples(samples_by_task_id, ids),
        metric=eval_metric,
        num_threads=num_threads,
        verbose=verbose,
        label="encdec GEPA optimized",
    )
    return optimized_program, baseline_scores, optimized_scores


def _failed_result_feedback(result: TestCaseResult) -> str:
    return "\n".join(
        [
            "First failing test:",
            f"input={json.dumps(result.input_value, default=str)}",
            f"expected={json.dumps(result.expected_output, default=str)}",
            f"actual={json.dumps(result.actual_output, default=str)}",
            f"error={result.error or ''}",
            f"compile_error={result.compile_error or ''}",
        ]
    )


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n..."
