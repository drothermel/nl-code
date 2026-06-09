from __future__ import annotations

from pathlib import Path
from typing import Any

import dspy
import pytest
from pydantic import BaseModel

from nl_code.code_execution.models import TestCaseResult
from nl_code.optim import humaneval_dspy_gepa as gepa_mod
from nl_code.optim.dspy_generators import EncoderDecoderCodeGenerator
from nl_code.optim.humaneval_dspy_gepa import (
    HumanEvalGepaFeedbackMetric,
    direct_gepa_metric,
)
from nl_code.optim.humaneval_dspy_optimize import (
    HumanEvalPassRateMetric,
    SplitTaskIds,
)


class FakeSample(BaseModel):
    task_id: str
    source__prompt: str
    gt_solution: str
    function_stub: str
    entry_point: str
    test_inputs: list[Any]
    test_results: list[Any] | None


def test_direct_gepa_metric_returns_feedback_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gepa_mod, "evaluate_completed_code", _fake_evaluate)
    metric = direct_gepa_metric(
        samples_by_task_id=_samples_by_task_id(),
        timeout_seconds=1.0,
        docker_image=None,
    )
    example = dspy.Example(task_id="HumanEval/0").with_inputs()
    prediction = dspy.Prediction(completed_code="def add_one(x):\n    return x + 1\n")

    score = metric(example, prediction, None, None, None)

    assert score.score == 0.5
    assert "pass_rate=0.500" in score.feedback
    assert "First failing test" in score.feedback


def test_encoder_gepa_path_uses_float_metric_for_eval_and_feedback_for_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[Any]] = {"evaluate": [], "compile": []}

    def fake_evaluate_splits(*, metric: Any, **kwargs: Any) -> dict[str, Any]:
        captured["evaluate"].append(metric)
        return {}

    def fake_compile_gepa(*, metric: Any, **kwargs: Any) -> Any:
        captured["compile"].append(metric)
        return object()

    monkeypatch.setattr(gepa_mod, "evaluate_splits", fake_evaluate_splits)
    monkeypatch.setattr(gepa_mod, "compile_gepa", fake_compile_gepa)
    monkeypatch.setattr(
        gepa_mod,
        "precompute_code_specs",
        lambda **_kwargs: {"HumanEval/0": "add one"},
    )

    baseline = EncoderDecoderCodeGenerator()
    gepa_mod._optimize_decoder_gepa(
        baseline=baseline,
        task_ids=SplitTaskIds(
            train=["HumanEval/0"],
            dev=["HumanEval/0"],
            eval=["HumanEval/0"],
        ),
        samples_by_task_id=_samples_by_task_id(),
        reflection_lm=object(),
        output_dir=Path("unused"),
        auto=None,
        max_metric_calls=None,
        num_threads=None,
        seed=0,
        timeout_seconds=1.0,
        docker_image=None,
        verbose=False,
    )

    assert len(captured["evaluate"]) == 2
    assert len(captured["compile"]) == 1
    assert isinstance(captured["evaluate"][0], HumanEvalPassRateMetric)
    assert isinstance(captured["compile"][0], HumanEvalGepaFeedbackMetric)


def test_direct_gepa_metric_raises_on_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nl_code.code_execution.models import CodeExecutionInfrastructureError

    def raise_infra(**_kwargs: Any) -> tuple[str, list[TestCaseResult], float]:
        raise CodeExecutionInfrastructureError(
            stage="run",
            execution_mode="function_call",
            detail="docker unavailable",
        )

    monkeypatch.setattr(gepa_mod, "evaluate_completed_code", raise_infra)
    metric = direct_gepa_metric(
        samples_by_task_id=_samples_by_task_id(),
        timeout_seconds=1.0,
        docker_image=None,
    )
    example = dspy.Example(task_id="HumanEval/0").with_inputs()
    prediction = dspy.Prediction(completed_code="def add_one(x):\n    return x + 1\n")

    with pytest.raises(CodeExecutionInfrastructureError, match="docker unavailable"):
        metric(example, prediction, None, None, None)


def test_direct_gepa_metric_handles_predictor_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gepa_mod, "evaluate_completed_code", _fake_evaluate)
    metric = direct_gepa_metric(
        samples_by_task_id=_samples_by_task_id(),
        timeout_seconds=1.0,
        docker_image=None,
    )
    example = dspy.Example(task_id="HumanEval/0").with_inputs()
    prediction = dspy.Prediction(completed_code="def add_one(x):\n    return x + 1\n")

    score = metric(example, prediction, None, "decoder.decode", None)

    assert score.score == 0.5
    assert "predictor=decoder.decode" in score.feedback
    assert "Use the code_spec" in score.feedback


def _samples_by_task_id() -> dict[str, FakeSample]:
    sample = FakeSample(
        task_id="HumanEval/0",
        source__prompt="def add_one(x):\n",
        gt_solution="def add_one(x):\n    return x + 1\n",
        function_stub="def add_one(x):\n",
        entry_point="add_one",
        test_inputs=[[1], [2]],
        test_results=[2, 4],
    )
    return {sample.task_id: sample}


def _fake_evaluate(**_kwargs: Any) -> tuple[str, list[TestCaseResult], float]:
    return (
        "def add_one(x):\n    return x + 1\n",
        [
            TestCaseResult(
                input_value=[1], expected_output=2, actual_output=2, passed=True
            ),
            TestCaseResult(
                input_value=[2], expected_output=4, actual_output=3, passed=False
            ),
        ],
        0.5,
    )
