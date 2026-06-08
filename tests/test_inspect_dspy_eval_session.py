from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from typer.testing import CliRunner


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "inspect_dspy_eval_session.py"


def test_build_package_session_report_preserves_forensic_details(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    session_dir = package_session(tmp_path)

    report = module.build_session_report(session_dir)

    assert report.session["session_id"] == "session_test"
    assert len(report.runs) == 1
    assert len(report.samples) == 1
    assert len(report.attempts) == 1
    assert len(report.generation_calls) == 1
    assert report.attempts[0].test_case_results[0].actual_output == 1
    assert report.attempts[0].generation_call_ids == [report.generation_calls[0].id]
    assert report.samples[0].best_passed is True
    assert report.aggregates["direct"].attempt_pass_rate == 1.0
    assert report.generation_calls[0].messages[0]["role"] == "system"


def test_build_legacy_session_report_preserves_outputs_and_failures(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    session_dir = legacy_session(tmp_path)

    report = module.build_session_report(session_dir)

    assert report.runs[0].run_format == module.RunFormat.LEGACY_NOTEBOOK
    assert report.attempts[0].task_id == "HumanEval/2"
    assert report.attempts[0].extracted_code == "def f(x):\n    return 0\n"
    assert report.attempts[0].failed_test_count == 1
    assert report.samples[0].best_passed is False
    assert report.aggregates["encdec"].average_test_pass_rate == 0.0


def test_cli_writes_output_file(tmp_path: Path) -> None:
    module = load_script_module()
    session_dir = package_session(tmp_path)
    output_path = tmp_path / "report.json"
    runner = CliRunner()

    result = runner.invoke(module.app, [str(session_dir), "-o", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    data = read_json(output_path)
    runs = cast(list[dict[str, Any]], data["runs"])
    assert data["schema_version"] == "dspy_eval_session_report_v0"
    assert runs[0]["id"] == "human_eval_dspy_run_20260515T000001Z"


def test_cli_walk_writes_one_report_per_eval_session(tmp_path: Path) -> None:
    module = load_script_module()
    corpus_dir = tmp_path / "v0"
    corpus_dir.mkdir()
    package_session(corpus_dir, session_id="session_000025")
    legacy_session(corpus_dir, session_id="session_000001")
    base_session(corpus_dir, session_id="session_000019")
    output_dir = tmp_path / "parsed"
    runner = CliRunner()

    result = runner.invoke(
        module.app,
        [str(corpus_dir), "--walk", "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0
    assert (output_dir / "session_000025.eval_report.json").exists()
    assert (output_dir / "session_000001.eval_report.json").exists()
    assert not (output_dir / "session_000019.eval_report.json").exists()
    manifest = read_json(output_dir / "manifest.json")
    reports = cast(list[dict[str, Any]], manifest["reports"])
    assert manifest["schema_version"] == "dspy_eval_session_report_manifest_v0"
    assert manifest["session_count"] == 3
    assert manifest["eval_session_count"] == 2
    assert manifest["skipped_session_count"] == 1
    assert {report["session_id"] for report in reports} == {
        "session_000001",
        "session_000025",
    }


def load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "inspect_dspy_eval_session",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load inspect_dspy_eval_session.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def package_session(tmp_path: Path, *, session_id: str = "session_test") -> Path:
    session_dir = base_session(tmp_path, session_id=session_id)
    logs_dir = session_dir / "raw" / "logs" / "eval_full_5x" / "baseline_direct"
    logs_dir.mkdir(parents=True)
    generation_log = logs_dir / "human_eval_dspy_generations_20260515T000000Z.jsonl"
    write_jsonl(generation_log, [generation_record()])
    write_json(
        logs_dir / "human_eval_dspy_run_20260515T000001Z.json",
        {
            "timestamp": "2026-05-15T00:00:01+00:00",
            "config": {
                "generation_type": "direct",
                "n_samples": 1,
                "num_repeats": 1,
                "model": "openrouter/test-model",
            },
            "selected_dataset_indices": [0],
            "attempts": [
                {
                    "generation_type": "direct",
                    "dataset_index": 0,
                    "task_id": "HumanEval/0",
                    "repeat_index": 0,
                    "skipped": False,
                    "error": None,
                    "code_spec": None,
                    "raw_completed_code": "def f(x):\n    return 1\n",
                    "extracted_code": "def f(x):\n    return 1\n",
                    "test_case_results": [passing_result()],
                    "test_pass_rate": 1.0,
                    "generation_log_file": (
                        "logs/eval_full_5x/baseline_direct/"
                        "human_eval_dspy_generations_20260515T000000Z.jsonl"
                    ),
                }
            ],
            "summaries": {
                "direct": {
                    "total_attempts": 1,
                    "evaluated_attempts": 1,
                    "skipped_count": 0,
                    "attempt_pass_count": 1,
                    "attempt_pass_rate": 1.0,
                    "sample_best_pass_count": 1,
                    "sample_best_pass_rate": 1.0,
                    "average_test_pass_rate": 1.0,
                }
            },
        },
    )
    return session_dir


def legacy_session(tmp_path: Path, *, session_id: str = "session_test") -> Path:
    session_dir = base_session(tmp_path, session_id=session_id)
    logs_dir = session_dir / "raw" / "logs"
    logs_dir.mkdir(parents=True)
    generation_log = logs_dir / "human_eval_dspy_20260515T000000Z.jsonl"
    write_jsonl(
        generation_log,
        [
            generation_record(
                generation_type="encdec",
                task_id="HumanEval/2",
                dataset_index=2,
            )
        ],
    )
    write_json(
        logs_dir / "human_eval_dspy_encdec_eval_20260515T000001Z.json",
        {
            "timestamp": "2026-05-15T00:00:01+00:00",
            "eval_type": "encdec",
            "val_num": 1,
            "seed": 42,
            "dataset_indices": [2],
            "outputs": [
                {
                    "dataset_index": 2,
                    "output": {
                        "skipped": False,
                        "error": None,
                        "task_id": "HumanEval/2",
                        "prediction": {
                            "code_spec": "return one",
                            "completed_code": "def f(x):\n    return 0\n",
                        },
                        "log_file": str(generation_log),
                        "extracted": "def f(x):\n    return 0\n",
                        "results": [failing_result()],
                        "pass_rate": 0.0,
                    },
                }
            ],
        },
    )
    return session_dir


def base_session(tmp_path: Path, *, session_id: str = "session_test") -> Path:
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    write_json(
        session_dir / "metadata.json",
        {
            "session_id": session_id,
            "session_kind": "eval_run",
            "confidence": "exact",
            "created_at": "2026-06-08T00:00:00Z",
            "source_roots": [],
            "original_grouping_paths": [],
            "extracted": {},
            "files": [],
        },
    )
    return session_dir


def generation_record(
    *,
    generation_type: str = "direct",
    task_id: str = "HumanEval/0",
    dataset_index: int = 0,
) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "Your input fields are:\n1. `code_stub` (str)\n",
            },
            {"role": "user", "content": "[[ ## code_stub ## ]]\ndef f(x):\n"},
        ],
        "outputs": ["def f(x):\n    return 1\n"],
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
            "generation_type": generation_type,
            "dataset_index": dataset_index,
            "task_id": task_id,
            "repeat_index": 0,
            "call_index": 0,
        },
    }


def passing_result() -> dict:
    return {
        "input_value": [1],
        "expected_output": 1,
        "actual_output": 1,
        "passed": True,
        "error": None,
        "compile_success": True,
        "compile_error": None,
    }


def failing_result() -> dict:
    return {
        "input_value": [1],
        "expected_output": 1,
        "actual_output": 0,
        "passed": False,
        "error": None,
        "compile_success": True,
        "compile_error": None,
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value
