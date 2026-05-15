from __future__ import annotations

from typing import Any

import dspy
import pytest
from pydantic import BaseModel

from nl_code.code_execution.models import TestCaseResult
from nl_code.optim import humaneval_dspy_optimize as opt_mod
from nl_code.optim.humaneval_dspy_optimize import (
    SplitTaskIds,
    direct_examples,
    encoder_examples,
    parse_task_ids,
    require_task_ids,
)


class FakeSample(BaseModel):
    task_id: str
    source__prompt: str
    gt_solution: str
    function_stub: str
    entry_point: str
    test_inputs: list[Any]
    test_results: list[Any] | None


def test_parse_task_ids_accepts_repeated_and_csv_values() -> None:
    assert parse_task_ids(["HumanEval/1, HumanEval/2", "HumanEval/3"]) == [
        "HumanEval/1",
        "HumanEval/2",
        "HumanEval/3",
    ]


def test_require_task_ids_rejects_empty_splits() -> None:
    with pytest.raises(ValueError, match="--dev-task-ids"):
        require_task_ids(SplitTaskIds(train=["HumanEval/1"], dev=[], eval=[]))


def test_direct_examples_only_pass_code_stub_as_program_input() -> None:
    examples = direct_examples(_samples_by_task_id(), ["HumanEval/0"])

    assert examples[0].inputs().toDict() == {"code_stub": "def add_one(x):\n"}
    assert examples[0].task_id == "HumanEval/0"
    assert examples[0].completed_code == "def add_one(x):\n    return x + 1\n"


def test_encoder_examples_only_pass_input_code_as_program_input() -> None:
    examples = encoder_examples(_samples_by_task_id(), ["HumanEval/0"])

    assert examples[0].inputs().toDict() == {
        "input_code": "def add_one(x):\n    return x + 1\n"
    }
    assert examples[0].function_stub == "def add_one(x):\n"


def test_metric_scores_generated_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(opt_mod, "evaluate_completed_code", _fake_evaluate)
    metric = opt_mod.direct_metric(
        samples_by_task_id=_samples_by_task_id(),
        timeout_seconds=1.0,
        docker_image=None,
        verbose=False,
        label="test",
    )
    example = dspy.Example(task_id="HumanEval/0").with_inputs()
    prediction = dspy.Prediction(completed_code="def add_one(x):\n    return x + 1\n")

    assert metric(example, prediction) == 1.0


def _samples_by_task_id() -> dict[str, FakeSample]:
    sample = FakeSample(
        task_id="HumanEval/0",
        source__prompt="def add_one(x):\n",
        gt_solution="def add_one(x):\n    return x + 1\n",
        function_stub="def add_one(x):\n",
        entry_point="add_one",
        test_inputs=[[1]],
        test_results=[2],
    )
    return {sample.task_id: sample}


def _fake_evaluate(**_kwargs: Any) -> tuple[str, list[TestCaseResult], float]:
    return "def add_one(x):\n    return x + 1\n", [], 1.0
