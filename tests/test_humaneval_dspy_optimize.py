from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import dspy
import pytest
from pydantic import BaseModel

from nl_code.code_execution.models import TestCaseResult
from nl_code.optim import dspy_generators as gen_mod
from nl_code.optim import humaneval_dspy_optimize as opt_mod
from nl_code.optim.dspy_generators import (
    resolve_openrouter_llm_config,
    supported_openrouter_llm_config_ids,
)
from nl_code.code_execution.models import CodeExecutionInfrastructureError
from nl_code.optim.humaneval_dspy_optimize import (
    OptimizationEventLogger,
    SplitTaskIds,
    api_key_from_env,
    direct_examples,
    encoder_examples,
    parse_task_ids,
    require_task_ids,
    score_value,
    validate_disjoint_splits,
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


def test_validate_disjoint_splits_rejects_overlapping_task_ids() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        validate_disjoint_splits(
            SplitTaskIds(
                train=["HumanEval/1"],
                dev=["HumanEval/1"],
                eval=["HumanEval/2"],
            )
        )


def test_validate_disjoint_splits_rejects_duplicate_ids_within_split() -> None:
    with pytest.raises(ValueError, match="duplicate task IDs in train"):
        validate_disjoint_splits(
            SplitTaskIds(
                train=["HumanEval/1", "HumanEval/1"],
                dev=["HumanEval/2"],
                eval=["HumanEval/3"],
            )
        )


def test_default_dspy_model_resolves_via_catalog() -> None:
    config = resolve_openrouter_llm_config(gen_mod.DEFAULT_DSPY_MODEL)
    assert config.model == "openrouter/openai/gpt-oss-20b"
    assert config.reasoning == {"effort": "low"}


def test_metric_raises_on_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_infra(**_kwargs: Any) -> tuple[str, list[TestCaseResult], float]:
        raise CodeExecutionInfrastructureError(
            stage="run",
            execution_mode="function_call",
            detail="docker unavailable",
        )

    monkeypatch.setattr(opt_mod, "evaluate_completed_code", raise_infra)
    metric = opt_mod.direct_metric(
        samples_by_task_id=_samples_by_task_id(),
        timeout_seconds=1.0,
        docker_image=None,
        verbose=False,
        label="test",
    )
    example = dspy.Example(task_id="HumanEval/0").with_inputs()
    prediction = dspy.Prediction(completed_code="def add_one(x):\n    return x + 1\n")

    with pytest.raises(CodeExecutionInfrastructureError, match="docker unavailable"):
        metric(example, prediction)


def test_nested_optimization_log_context_restores_event_logger(
    tmp_path: Path,
) -> None:
    outer_events = tmp_path / "outer.jsonl"
    inner_events = tmp_path / "inner.jsonl"
    outer_logger = OptimizationEventLogger(outer_events)
    inner_logger = OptimizationEventLogger(inner_events)

    with outer_logger:
        token = opt_mod._event_logger_var.set(outer_logger)
        try:
            with inner_logger:
                inner_token = opt_mod._event_logger_var.set(inner_logger)
                try:
                    opt_mod.log_optimization_event("inner_event")
                finally:
                    opt_mod._event_logger_var.reset(inner_token)
            opt_mod.log_optimization_event("outer_event")
        finally:
            opt_mod._event_logger_var.reset(token)

    assert "inner_event" in inner_events.read_text()
    assert "outer_event" in outer_events.read_text()
    assert "inner_event" not in outer_events.read_text()


def test_api_key_from_env_rejects_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "...")

    with pytest.raises(ValueError, match="placeholder"):
        api_key_from_env()


def test_normalize_auto_rejects_manual_mode() -> None:
    with pytest.raises(ValueError, match="manual MIPRO settings"):
        opt_mod.normalize_auto("none")


def test_score_value_accepts_score_with_feedback_like_object() -> None:
    class Score:
        score = 0.75

    assert score_value(Score()) == 0.75
    assert score_value({"score": 0.5, "feedback": "ok"}) == 0.5
    assert score_value(1.0) == 1.0


def test_supported_openrouter_llm_config_ids_are_low_off_or_na() -> None:
    assert supported_openrouter_llm_config_ids() == (
        "openrouter/openai/gpt-oss-20b/low/v1",
        "openrouter/deepseek/deepseek-chat-v3.1/off/v1",
        "openrouter/xiaomi/mimo-v2-flash/off/v1",
        "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1",
        "openrouter/baidu/ernie-4.5-21b-a3b/na/v1",
        "openrouter/bytedance-seed/seed-2.0-mini/off/v1",
        "openrouter/mistralai/devstral-small/na/v1",
        "openrouter/meta-llama/llama-4-scout/na/v1",
        "openrouter/qwen/qwen3-coder-30b-a3b-instruct/na/v1",
    )


def test_openrouter_llm_config_reasoning_variants() -> None:
    low = resolve_openrouter_llm_config("openrouter/openai/gpt-oss-20b/low/v1")
    off = resolve_openrouter_llm_config("openrouter/deepseek/deepseek-chat-v3.1/off/v1")
    no_control = resolve_openrouter_llm_config(
        "openrouter/mistralai/devstral-small/na/v1"
    )

    assert low.model == "openrouter/openai/gpt-oss-20b"
    assert low.reasoning == {"effort": "low"}
    assert off.model == "openrouter/deepseek/deepseek-chat-v3.1"
    assert off.reasoning == {"enabled": False}
    assert no_control.model == "openrouter/mistralai/devstral-small"
    assert no_control.reasoning is None


def test_configure_dspy_lm_prefers_explicit_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeLm:
        def __init__(self, model: str, **kwargs: Any) -> None:
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch.setattr(gen_mod.dspy, "LM", FakeLm)
    monkeypatch.setattr(gen_mod.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(gen_mod.dspy, "configure_cache", lambda **kwargs: None)

    gen_mod.configure_dspy_lm(
        model="openrouter/deepseek/deepseek-chat-v3.1",
        api_key="test-key",
        reasoning_effort="minimal",
        reasoning={"enabled": False},
    )

    assert captured["model"] == "openrouter/deepseek/deepseek-chat-v3.1"
    assert captured["kwargs"]["reasoning"] == {"enabled": False}


def test_configure_dspy_lm_can_omit_catalog_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeLm:
        def __init__(self, model: str, **kwargs: Any) -> None:
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch.setattr(gen_mod.dspy, "LM", FakeLm)
    monkeypatch.setattr(gen_mod.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(gen_mod.dspy, "configure_cache", lambda **kwargs: None)

    gen_mod.configure_dspy_lm(
        model="openrouter/mistralai/devstral-small",
        api_key="test-key",
        reasoning_effort=None,
        reasoning=None,
    )

    assert captured["model"] == "openrouter/mistralai/devstral-small"
    assert captured["kwargs"]["reasoning"] is None


def test_optimization_artifact_paths_use_shared_namespace(tmp_path: Path) -> None:
    artifacts = opt_mod.optimization_artifact_paths(
        output_dir=tmp_path,
        generation_type="direct",
        optimization_target=None,
        timestamp="20260515T000001Z",
    )

    assert artifacts.stem == "human_eval_dspy_direct_optimized_20260515T000001Z"
    assert artifacts.optimized_program_path == tmp_path / f"{artifacts.stem}.json"
    assert artifacts.summary_path == tmp_path / f"{artifacts.stem}_summary.json"
    assert artifacts.run_log_path == tmp_path / f"{artifacts.stem}_run.log"
    assert artifacts.event_log_path == tmp_path / f"{artifacts.stem}_events.jsonl"


def test_optimization_log_context_tees_output_and_events(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_log_path = tmp_path / "run.log"
    event_log_path = tmp_path / "events.jsonl"

    with opt_mod.optimization_log_context(
        run_log_path=run_log_path,
        event_log_path=event_log_path,
    ):
        print("stdout line")
        print("stderr line", file=sys.stderr)
        opt_mod.log_step("structured step", verbose=False)

    captured = capsys.readouterr()
    assert "stdout line" in captured.out
    assert "stderr line" in captured.err
    run_log = run_log_path.read_text()
    assert "stdout line" in run_log
    assert "stderr line" in run_log
    events = [json.loads(line) for line in event_log_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "run_start",
        "step",
        "run_end",
    ]
    assert events[1]["payload"] == {"message": "structured step"}


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


def test_metric_accepts_bootstrap_trace_argument(
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

    assert metric(example, prediction, []) == 1.0


def test_encoder_metric_accepts_bootstrap_trace_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(opt_mod, "evaluate_completed_code", _fake_evaluate)

    class FakeDecoder:
        def __call__(self, *, code_spec: str, function_stub: str) -> dspy.Prediction:
            return dspy.Prediction(
                completed_code=f"{function_stub}    return {code_spec}\n"
            )

    metric = opt_mod.encoder_metric(
        decoder=FakeDecoder(),
        samples_by_task_id=_samples_by_task_id(),
        timeout_seconds=1.0,
        docker_image=None,
        verbose=False,
        label="test",
    )
    example = dspy.Example(
        task_id="HumanEval/0",
        function_stub="def add_one(x):\n",
    ).with_inputs("input_code")
    prediction = dspy.Prediction(code_spec="x + 1")

    assert metric(example, prediction, []) == 1.0


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
