from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from nl_code.code_execution.models import TestCaseResult
from nl_code.optim import humaneval_dspy_eval as eval_mod
from nl_code.optim.humaneval_dspy_eval import (
    GenerationType,
    HumanEvalDspyAttemptResult,
    HumanEvalDspyEvalConfig,
    log_eval_progress,
    select_dataset_indices,
    summarize_attempts,
)


class FakeSample(BaseModel):
    task_id: str
    source__prompt: str
    gt_solution: str
    function_stub: str
    entry_point: str
    test_inputs: list[Any]
    test_results: list[Any] | None


class FakeDataset:
    def __init__(self, samples: list[FakeSample]) -> None:
        self.raw_samples = {sample.task_id: sample for sample in samples}

    def get_raw_sample_at_index(self, index: int) -> FakeSample:
        return list(self.raw_samples.values())[index]


class FakePrediction(BaseModel):
    completed_code: str
    code_spec: str | None = None


class FakeDirectGenerator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, *, code_stub: str) -> FakePrediction:
        self.calls.append(code_stub)
        return FakePrediction(completed_code="def add_one(x):\n    return x + 1\n")


class FakeEncoderDecoderGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, *, input_code: str, function_stub: str) -> FakePrediction:
        self.calls.append((input_code, function_stub))
        return FakePrediction(
            code_spec="add one",
            completed_code="```python\ndef add_one(x):\n    return x + 1\n```",
        )


class FakeLm:
    history: list[dict[str, Any]] = [{"model": "fake"}]


class MutableFakeLm:
    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []


class FakeLoadableProgram:
    def __init__(self) -> None:
        self.loaded_path: Path | None = None

    def load(self, path: Path) -> None:
        self.loaded_path = path


class FakeEncDecProgram(FakeLoadableProgram):
    def __init__(
        self,
        encoder: FakeLoadableProgram | None = None,
        decoder: FakeLoadableProgram | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def test_select_dataset_indices_skips_unevaluable_samples() -> None:
    dataset = _fake_dataset()

    selected = select_dataset_indices(dataset, n_samples=3, seed=42)

    assert set(selected) == {0, 2}


def test_select_dataset_indices_by_task_id_preserves_order() -> None:
    dataset = _fake_dataset()

    selected = select_dataset_indices(
        dataset,
        n_samples=1,
        seed=42,
        task_ids=["HumanEval/2", "HumanEval/0"],
    )

    assert selected == [2, 0]


def test_config_rejects_mixed_selection_modes() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        HumanEvalDspyEvalConfig(
            sample_indices=[0],
            task_ids=["HumanEval/0"],
        )


def test_config_rejects_explicit_selection_with_nondefault_sample_count() -> None:
    with pytest.raises(ValueError, match="n_samples cannot be combined"):
        HumanEvalDspyEvalConfig(sample_indices=[0], n_samples=2)


def test_config_rejects_nonpositive_repeats() -> None:
    with pytest.raises(ValueError, match="num_repeats must be positive"):
        HumanEvalDspyEvalConfig(num_repeats=0)


def test_config_rejects_negative_progress_interval() -> None:
    with pytest.raises(ValueError, match="log_every must be non-negative"):
        HumanEvalDspyEvalConfig(log_every=-1)


def test_config_rejects_full_encdec_and_component_programs() -> None:
    with pytest.raises(ValueError, match="encdec_program_path cannot be combined"):
        HumanEvalDspyEvalConfig(
            encdec_program_path=Path("encdec.json"),
            encoder_program_path=Path("encoder.json"),
        )


def test_load_direct_generator_loads_configured_program(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(eval_mod, "DirectCodeGenerator", FakeLoadableProgram)
    program_path = tmp_path / "direct.json"

    generator = eval_mod.load_direct_generator(program_path)

    assert isinstance(generator, FakeLoadableProgram)
    assert generator.loaded_path == program_path


def test_load_encoder_decoder_generator_composes_component_programs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(eval_mod, "CodeSpecEncoder", FakeLoadableProgram)
    monkeypatch.setattr(eval_mod, "CodeSpecDecoder", FakeLoadableProgram)
    monkeypatch.setattr(eval_mod, "EncoderDecoderCodeGenerator", FakeEncDecProgram)
    encoder_path = tmp_path / "encoder.json"
    decoder_path = tmp_path / "decoder.json"

    generator = eval_mod.load_encoder_decoder_generator(
        encoder_program_path=encoder_path,
        decoder_program_path=decoder_path,
    )

    assert isinstance(generator, FakeEncDecProgram)
    assert generator.encoder.loaded_path == encoder_path
    assert generator.decoder.loaded_path == decoder_path


def test_load_encoder_decoder_generator_loads_full_program(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(eval_mod, "EncoderDecoderCodeGenerator", FakeEncDecProgram)
    program_path = tmp_path / "encdec.json"

    generator = eval_mod.load_encoder_decoder_generator(
        encdec_program_path=program_path,
        encoder_program_path=None,
        decoder_program_path=None,
    )

    assert isinstance(generator, FakeEncDecProgram)
    assert generator.loaded_path == program_path
    assert generator.encoder is None
    assert generator.decoder is None


def test_run_direct_eval_expands_repeats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(eval_mod, "run_test_cases", _fake_run_test_cases)
    dataset = _fake_dataset()
    direct_generator = FakeDirectGenerator()

    run = eval_mod.run_humaneval_dspy_eval(
        HumanEvalDspyEvalConfig(
            generation_type=GenerationType.DIRECT,
            sample_indices=[0],
            num_repeats=2,
            output_dir=tmp_path,
        ),
        dataset=dataset,
        direct_generator=direct_generator,
        lm=FakeLm(),
    )

    assert [attempt.repeat_index for attempt in run.attempts] == [0, 1]
    assert [attempt.generation_type for attempt in run.attempts] == [
        GenerationType.DIRECT,
        GenerationType.DIRECT,
    ]
    assert direct_generator.calls == [
        dataset.get_raw_sample_at_index(0).source__prompt,
        dataset.get_raw_sample_at_index(0).source__prompt,
    ]
    assert run.summaries["direct"].attempt_pass_rate == 1.0
    assert run.run_log_file is not None
    assert run.run_log_file.is_file()


def test_run_both_eval_uses_same_selected_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(eval_mod, "run_test_cases", _fake_run_test_cases)
    direct_generator = FakeDirectGenerator()
    encoder_decoder_generator = FakeEncoderDecoderGenerator()

    run = eval_mod.run_humaneval_dspy_eval(
        HumanEvalDspyEvalConfig(
            generation_type=GenerationType.BOTH,
            sample_indices=[2],
            output_dir=tmp_path,
        ),
        dataset=_fake_dataset(),
        direct_generator=direct_generator,
        encoder_decoder_generator=encoder_decoder_generator,
        lm=FakeLm(),
    )

    assert run.selected_dataset_indices == [2]
    assert [attempt.generation_type for attempt in run.attempts] == [
        GenerationType.DIRECT,
        GenerationType.ENCDEC,
    ]
    assert set(run.summaries) == {"direct", "encdec"}
    assert len(direct_generator.calls) == 1
    assert encoder_decoder_generator.calls == [
        ("def add_one(x):\n", "def add_one(x):\n")
    ]


def test_encdec_eval_oracle_input_uses_gt_solution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(eval_mod, "run_test_cases", _fake_run_test_cases)
    encoder_decoder_generator = FakeEncoderDecoderGenerator()
    dataset = _fake_dataset()

    eval_mod.run_humaneval_dspy_eval(
        HumanEvalDspyEvalConfig(
            generation_type=GenerationType.ENCDEC,
            sample_indices=[0],
            encoder_input="oracle",
            output_dir=tmp_path,
        ),
        dataset=dataset,
        direct_generator=FakeDirectGenerator(),
        encoder_decoder_generator=encoder_decoder_generator,
        lm=FakeLm(),
    )

    sample = dataset.get_raw_sample_at_index(0)
    assert encoder_decoder_generator.calls == [
        (sample.gt_solution, sample.function_stub)
    ]


def test_eval_config_resolves_llm_catalog_id() -> None:
    config = HumanEvalDspyEvalConfig(
        llm_config_id="openrouter/openai/gpt-oss-20b/low/v1",
        model="unused",
        reasoning_effort="minimal",
        reasoning_config={"effort": "low"},
    )

    assert config.llm_config_id == "openrouter/openai/gpt-oss-20b/low/v1"
    assert config.reasoning_config == {"effort": "low"}


def test_fenced_code_is_extracted_before_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_code = ""

    def capture_run_test_cases(**kwargs: Any) -> tuple[list[TestCaseResult], float]:
        nonlocal captured_code
        captured_code = kwargs["code"]
        return _fake_run_test_cases(**kwargs)

    monkeypatch.setattr(eval_mod, "run_test_cases", capture_run_test_cases)

    run = eval_mod.run_humaneval_dspy_eval(
        HumanEvalDspyEvalConfig(
            generation_type=GenerationType.ENCDEC,
            sample_indices=[0],
            output_dir=tmp_path,
        ),
        dataset=_fake_dataset(),
        direct_generator=FakeDirectGenerator(),
        encoder_decoder_generator=FakeEncoderDecoderGenerator(),
        lm=FakeLm(),
    )

    assert run.attempts[0].extracted_code == "def add_one(x):\n    return x + 1\n"
    assert "```" not in captured_code


def test_encdec_eval_logs_all_lm_calls_from_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(eval_mod, "run_test_cases", _fake_run_test_cases)
    lm = MutableFakeLm()
    generation_log_file = tmp_path / "generations.jsonl"

    class RecordingEncoderDecoderGenerator:
        def __call__(self, *, input_code: str, function_stub: str) -> FakePrediction:
            lm.history.append(_history_record("encode", input_code))
            lm.history.append(_history_record("decode", function_stub))
            return FakePrediction(
                code_spec="add one",
                completed_code="def add_one(x):\n    return x + 1\n",
            )

    run = eval_mod.run_humaneval_dspy_eval(
        HumanEvalDspyEvalConfig(
            generation_type=GenerationType.ENCDEC,
            sample_indices=[0],
            output_dir=tmp_path,
            generation_log_file=generation_log_file,
        ),
        dataset=_fake_dataset(),
        direct_generator=FakeDirectGenerator(),
        encoder_decoder_generator=RecordingEncoderDecoderGenerator(),
        lm=lm,
    )

    records = [
        json.loads(line)
        for line in generation_log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert run.attempts[0].generation_log_file == generation_log_file
    assert [record["kind"] for record in records] == ["encode", "decode"]
    assert [record["attempt"]["call_index"] for record in records] == [0, 1]
    assert {record["attempt"]["task_id"] for record in records} == {"HumanEval/0"}
    assert {record["attempt"]["generation_type"] for record in records} == {"encdec"}


def test_summary_reports_attempt_and_best_of_n_rates() -> None:
    summary = summarize_attempts(
        [
            _attempt(dataset_index=0, repeat_index=0, pass_rate=0.0),
            _attempt(dataset_index=0, repeat_index=1, pass_rate=1.0),
            _attempt(dataset_index=1, repeat_index=0, pass_rate=0.5),
            _attempt(dataset_index=1, repeat_index=1, pass_rate=0.0),
        ]
    )

    assert summary.total_attempts == 4
    assert summary.attempt_pass_rate == 0.25
    assert summary.sample_best_pass_rate == 0.5
    assert summary.average_test_pass_rate == 0.375


def test_progress_logging_respects_interval(capsys: pytest.CaptureFixture[str]) -> None:
    log_eval_progress(completed_attempts=1, total_attempts=3, log_every=2)
    assert capsys.readouterr().out == ""

    log_eval_progress(completed_attempts=2, total_attempts=3, log_every=2)
    interval_output = capsys.readouterr().out
    assert "completed 2/3 (66.7%)" in interval_output

    log_eval_progress(completed_attempts=3, total_attempts=3, log_every=2)
    final_output = capsys.readouterr().out
    assert "completed 3/3 (100.0%)" in final_output


def test_progress_logging_can_be_disabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_eval_progress(completed_attempts=1, total_attempts=1, log_every=0)

    assert capsys.readouterr().out == ""


def test_run_log_serializes_repeated_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(eval_mod, "run_test_cases", _fake_run_test_cases)
    run_log_file = tmp_path / "run.json"

    run = eval_mod.run_humaneval_dspy_eval(
        HumanEvalDspyEvalConfig(
            sample_indices=[0],
            num_repeats=2,
            output_dir=tmp_path,
            run_log_file=run_log_file,
        ),
        dataset=_fake_dataset(),
        direct_generator=FakeDirectGenerator(),
        encoder_decoder_generator=FakeEncoderDecoderGenerator(),
        lm=FakeLm(),
    )

    payload = json.loads(run_log_file.read_text())
    assert run.run_log_file == run_log_file
    assert [attempt["repeat_index"] for attempt in payload["attempts"]] == [0, 1]


def _fake_dataset() -> FakeDataset:
    return FakeDataset(
        [
            FakeSample(
                task_id="HumanEval/0",
                source__prompt="def add_one(x):\n",
                gt_solution="def add_one(x):\n    return x + 1\n",
                function_stub="def add_one(x):\n",
                entry_point="add_one",
                test_inputs=[[1]],
                test_results=[2],
            ),
            FakeSample(
                task_id="HumanEval/1",
                source__prompt="def skip(x):\n",
                gt_solution="def skip(x):\n    return x\n",
                function_stub="def skip(x):\n",
                entry_point="skip",
                test_inputs=[[1]],
                test_results=None,
            ),
            FakeSample(
                task_id="HumanEval/2",
                source__prompt="def add_one(x):\n",
                gt_solution="def add_one(x):\n    return x + 1\n",
                function_stub="def add_one(x):\n",
                entry_point="add_one",
                test_inputs=[[4]],
                test_results=[5],
            ),
        ]
    )


def _fake_run_test_cases(**kwargs: Any) -> tuple[list[TestCaseResult], float]:
    test_cases = kwargs["test_cases"]
    return [
        TestCaseResult(
            input_value=test_case.input_value,
            expected_output=test_case.expected_output,
            actual_output=test_case.expected_output,
            passed=True,
        )
        for test_case in test_cases
    ], 1.0


def _history_record(kind: str, content: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "messages": [{"role": "user", "content": content}],
        "outputs": [content],
        "usage": {"total_tokens": 1},
    }


def _attempt(
    *,
    dataset_index: int,
    repeat_index: int,
    pass_rate: float,
) -> HumanEvalDspyAttemptResult:
    return HumanEvalDspyAttemptResult(
        generation_type=GenerationType.DIRECT,
        dataset_index=dataset_index,
        task_id=f"HumanEval/{dataset_index}",
        repeat_index=repeat_index,
        test_pass_rate=pass_rate,
    )
