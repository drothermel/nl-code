from __future__ import annotations

import json
from pathlib import Path

from nl_code.optim.humaneval_dspy_logs import (
    load_humaneval_dspy_log_snapshot,
    parse_humaneval_dspy_logs,
    write_humaneval_dspy_log_snapshot,
)


def test_parse_legacy_eval_log_and_generation_history(tmp_path: Path) -> None:
    generation_log = tmp_path / "human_eval_dspy_20260515T000000Z.jsonl"
    generation_log.write_text(json.dumps(_generation_record()) + "\n")
    run_log = tmp_path / "human_eval_dspy_direct_eval_20260515T000001Z.json"
    run_log.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-15T00:00:01+00:00",
                "eval_type": "direct",
                "val_num": 1,
                "seed": 42,
                "dataset_indices": [0],
                "outputs": [
                    {
                        "dataset_index": 0,
                        "output": {
                            "skipped": False,
                            "error": None,
                            "task_id": "HumanEval/0",
                            "prediction": {
                                "completed_code": "def f():\n    return 1\n"
                            },
                            "log_file": str(generation_log),
                            "extracted": "def f():\n    return 1\n",
                            "results": [_passing_result()],
                            "pass_rate": 1.0,
                        },
                    }
                ],
            }
        )
    )

    snapshot = parse_humaneval_dspy_logs(tmp_path)

    assert len(snapshot.pipelines) == 1
    assert snapshot.pipelines[0].generation_type == "direct"
    assert len(snapshot.runs) == 1
    assert snapshot.runs[0].generation_call_count == 1
    assert snapshot.attempts[0].task_id == "HumanEval/0"
    assert snapshot.attempts[0].passed is True
    attempt_calls = snapshot.generation_calls_for_attempt(snapshot.attempts[0])
    assert attempt_calls[0].attempt is not None
    assert [call.attempt.call_index for call in attempt_calls if call.attempt] == [0]


def test_parse_package_run_log_round_trips_snapshot(tmp_path: Path) -> None:
    run_log = tmp_path / "human_eval_dspy_run_20260515T000001Z.json"
    run_log.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-15T00:00:01+00:00",
                "config": {
                    "generation_type": "encdec",
                    "n_samples": 1,
                    "seed": 7,
                },
                "selected_dataset_indices": [2],
                "attempts": [
                    {
                        "generation_type": "encdec",
                        "dataset_index": 2,
                        "task_id": "HumanEval/2",
                        "repeat_index": 0,
                        "skipped": False,
                        "error": "wrong answer",
                        "code_spec": "return one",
                        "raw_completed_code": "def f():\n    return 0\n",
                        "extracted_code": "def f():\n    return 0\n",
                        "test_case_results": [_failing_result()],
                        "test_pass_rate": 0.0,
                        "generation_log_file": None,
                    }
                ],
                "summaries": {
                    "encdec": {
                        "total_attempts": 1,
                        "evaluated_attempts": 1,
                        "skipped_count": 0,
                        "attempt_pass_count": 0,
                        "attempt_pass_rate": 0.0,
                        "sample_best_pass_count": 0,
                        "sample_best_pass_rate": 0.0,
                        "average_test_pass_rate": 0.0,
                    }
                },
            }
        )
    )
    snapshot_path = tmp_path / "snapshot.json"

    snapshot = parse_humaneval_dspy_logs(tmp_path)
    write_humaneval_dspy_log_snapshot(snapshot, snapshot_path)
    loaded = load_humaneval_dspy_log_snapshot(snapshot_path)

    assert loaded.failed_task_ids == ["HumanEval/2"]
    assert loaded.runs[0].stats_by_generation_type["encdec"].attempt_pass_rate == 0.0
    assert loaded.attempts_for_task("HumanEval/2")[0].code_spec == "return one"


def _generation_record() -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "Your input fields are:\n1. `code_stub` (str)\n",
            },
            {"role": "user", "content": "[[ ## code_stub ## ]]\ndef f():\n"},
        ],
        "outputs": ["def f():\n    return 1\n"],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "cost": 0.1,
        "timestamp": "2026-05-15T00:00:00+00:00",
        "uuid": "call-1",
        "model": "openrouter/test-model",
        "response_model": "test-model",
        "model_type": "chat",
        "attempt": {
            "generation_type": "direct",
            "dataset_index": 0,
            "task_id": "HumanEval/0",
            "repeat_index": 0,
            "call_index": 0,
        },
    }


def _passing_result() -> dict:
    return {
        "input_value": [1],
        "expected_output": 1,
        "actual_output": 1,
        "passed": True,
        "error": None,
        "compile_success": True,
        "compile_error": None,
    }


def _failing_result() -> dict:
    return {
        "input_value": [1],
        "expected_output": 1,
        "actual_output": 0,
        "passed": False,
        "error": None,
        "compile_success": True,
        "compile_error": None,
    }
