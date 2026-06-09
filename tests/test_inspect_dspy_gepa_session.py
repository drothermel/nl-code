from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from typer.testing import CliRunner


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "inspect_dspy_gepa_session.py"


def test_build_gepa_session_report_preserves_forensic_details(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    session_dir = gepa_session(tmp_path)

    report = module.build_session_report(session_dir)

    assert report.session["session_id"] == "session_gepa"
    assert len(report.optimizer_runs) == 1
    assert report.optimizer_runs[0].generation_type == "direct_gepa"
    assert len(report.programs) == 3
    assert report.programs[1].signature_instructions == "Optimized instruction"
    assert report.programs[1].signature_fields[0]["prefix"] == "Code Stub:"
    assert report.programs[2].phase == module.ProgramPhase.CANDIDATE
    assert report.programs[2].signature_instructions == (
        "Better instruction\nwith continuation"
    )
    assert report.programs[2].parent_program_id.endswith(":program:baseline")
    assert len(report.split_evaluations) == 4
    assert len(report.task_scores) == 4
    assert report.task_scores[1].task_id == "HumanEval/1"
    assert report.task_scores[1].pass_rate == 0.5
    assert len(report.metric_calls) == 3
    assert report.metric_calls[0].phase == "baseline"
    assert report.metric_calls[0].split == "train"
    assert report.metric_calls[2].kind == module.MetricCallKind.GEPA
    assert len(report.generated_outputs) == 1
    assert report.generated_outputs[0].task_id == "HumanEval/1"
    assert report.generated_outputs[0].completed_code == "def g():\n    return 1\n"
    assert len(report.optimizer_iterations) == 1
    iteration = report.optimizer_iterations[0]
    assert iteration.iteration == 1
    assert iteration.selected_program_index == 0
    assert iteration.proposed_predictor == "complete"
    assert iteration.new_candidate_index == 1
    assert iteration.individual_valset_scores == {0: 1.0}
    assert len(report.best_effort.candidate_rankings) == 1
    assert len(report.best_effort.candidate_proposals) == 1
    assert len(report.state_files) == 1
    assert report.state_files[0].decoded is False
    assert report.state_files[0].pickle_protocol is not None
    assert "program_candidates" in report.state_files[0].string_keys_seen
    assert report.aggregates.optimizer_iteration_count == 1
    assert any(
        "not_present: historical GEPA artifacts" in note for note in report.parse_notes
    )


def test_cli_writes_output_file(tmp_path: Path) -> None:
    module = load_script_module()
    session_dir = gepa_session(tmp_path)
    output_path = tmp_path / "report.json"
    runner = CliRunner()

    result = runner.invoke(module.app, [str(session_dir), "-o", str(output_path)])

    assert result.exit_code == 0
    data = read_json(output_path)
    assert data["schema_version"] == "dspy_gepa_session_report_v0"
    runs = cast(list[dict[str, Any]], data["optimizer_runs"])
    assert runs[0]["id"] == "human_eval_dspy_direct_gepa_optimized_20260515T000000Z"


def test_cli_walk_writes_one_report_per_gepa_session(tmp_path: Path) -> None:
    module = load_script_module()
    corpus_dir = tmp_path / "v0"
    corpus_dir.mkdir()
    gepa_session(corpus_dir, session_id="session_000018")
    base_session(corpus_dir, session_id="session_000025", session_kind="eval_run")
    output_dir = tmp_path / "parsed"
    runner = CliRunner()

    result = runner.invoke(
        module.app,
        [str(corpus_dir), "--walk", "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0
    assert (output_dir / "session_000018.gepa_report.json").exists()
    assert not (output_dir / "session_000025.gepa_report.json").exists()
    manifest = read_json(output_dir / "manifest.json")
    reports = cast(list[dict[str, Any]], manifest["reports"])
    assert manifest["schema_version"] == "dspy_gepa_session_report_manifest_v0"
    assert manifest["session_count"] == 2
    assert manifest["gepa_session_count"] == 1
    assert manifest["skipped_session_count"] == 1
    assert reports[0]["session_id"] == "session_000018"


def load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "inspect_dspy_gepa_session",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load inspect_dspy_gepa_session.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gepa_session(tmp_path: Path, *, session_id: str = "session_gepa") -> Path:
    session_dir = base_session(
        tmp_path, session_id=session_id, session_kind="optimization"
    )
    logs_dir = session_dir / "raw" / "logs" / "dspy_optimized"
    logs_dir.mkdir(parents=True)
    stem = "human_eval_dspy_direct_gepa_optimized_20260515T000000Z"
    write_json(logs_dir / f"{stem}_summary.json", summary_payload(stem))
    write_json(logs_dir / f"{stem}.json", program_payload())
    write_jsonl(
        logs_dir / f"{stem}_events.jsonl",
        [
            {
                "timestamp": "2026-05-15T00:00:00Z",
                "event": "metric_call",
                "payload": {
                    "label": "direct/gepa-eval",
                    "metric_call": 1,
                    "task_id": "HumanEval/0",
                    "pass_rate": 1.0,
                    "error": None,
                },
            },
            {
                "timestamp": "2026-05-15T00:00:01Z",
                "event": "metric_call",
                "payload": {
                    "label": "direct/gepa-eval",
                    "metric_call": 2,
                    "task_id": "HumanEval/1",
                    "pass_rate": 0.5,
                    "error": None,
                },
            },
            {
                "timestamp": "2026-05-15T00:00:02Z",
                "event": "gepa_metric_call",
                "payload": {
                    "label": "direct-gepa",
                    "metric_call": 1,
                    "task_id": "HumanEval/1",
                    "predictor": "complete",
                    "pass_rate": 1.0,
                    "error": None,
                },
            },
        ],
    )
    run_log_path = logs_dir / f"{stem}_run.log"
    run_log_path.write_text(
        "\n".join(
            [
                "2026-05-15T00:00:00+00:00 Starting GEPA optimization auto=None max_metric_calls=3 train=1 dev=1 num_threads=1 log_dir=logs/dspy_optimized/gepa_logs/20260515T000000Z",
                "2026/05/15 00:00:01 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Selected program 0 score: 0.5",
                "2026/05/15 00:00:02 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Proposed new text for complete: Better instruction",
                "with continuation",
                "2026/05/15 00:00:03 INFO dspy.teleprompt.gepa.gepa: Iteration 1: New subsample score 1.0 is better than old score 0.5. Continue to full eval and add to candidate pool.",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Found a better program on the valset with score 1.0",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Valset score for new program: 1.0 (coverage 1 / 1)",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Individual valset scores for new program: {0: 1.0}",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: New valset pareto front scores: {0: 1.0}",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Updated valset pareto front programs: {0: {0, 1}}",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Valset pareto front aggregate score: 1.0",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best program as per aggregate score on valset: 1",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Best score on valset: 1.0",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: Linear pareto front program index: 1",
                "2026/05/15 00:00:04 INFO dspy.teleprompt.gepa.gepa: Iteration 1: New program candidate index: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gepa_dir = logs_dir / "gepa_logs" / "20260515T000000Z"
    output_dir = gepa_dir / "generated_best_outputs_valset" / "task_0"
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "iter_0_prog_0.json",
        {"completed_code": "def g():\n    return 1\n"},
    )
    with (gepa_dir / "gepa_state.bin").open("wb") as handle:
        pickle.dump(
            {
                "program_candidates": [{"complete": "instruction"}],
                "prog_candidate_val_subscores": [{0: 1.0}],
            },
            handle,
            protocol=4,
        )
    return session_dir


def base_session(
    tmp_path: Path,
    *,
    session_id: str,
    session_kind: str,
) -> Path:
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    write_json(
        session_dir / "metadata.json",
        {
            "session_id": session_id,
            "session_kind": session_kind,
            "confidence": "exact",
            "created_at": "2026-06-08T00:00:00Z",
            "source_roots": [],
            "original_grouping_paths": [],
            "extracted": {},
            "files": [],
        },
    )
    return session_dir


def summary_payload(stem: str) -> dict[str, Any]:
    return {
        "timestamp": "2026-05-15T00:00:05Z",
        "generation_type": "direct_gepa",
        "optimization_target": None,
        "model": "openrouter/test-model",
        "auto": None,
        "max_metric_calls": 3,
        "num_threads": 1,
        "seed": 42,
        "train_task_ids": ["HumanEval/0"],
        "dev_task_ids": ["HumanEval/1"],
        "eval_task_ids": [],
        "baseline_scores": {
            "train": split_score("direct GEPA baseline/train", {"HumanEval/0": 1.0}),
            "dev": split_score("direct GEPA baseline/dev", {"HumanEval/1": 0.5}),
        },
        "optimized_scores": {
            "train": split_score("direct GEPA optimized/train", {"HumanEval/0": 1.0}),
            "dev": split_score("direct GEPA optimized/dev", {"HumanEval/1": 1.0}),
        },
        "optimized_program_path": f"logs/dspy_optimized/{stem}.json",
        "summary_path": f"logs/dspy_optimized/{stem}_summary.json",
        "run_log_path": f"logs/dspy_optimized/{stem}_run.log",
        "event_log_path": f"logs/dspy_optimized/{stem}_events.jsonl",
    }


def split_score(split_name: str, task_scores: dict[str, float]) -> dict[str, Any]:
    values = list(task_scores.values())
    task_count = len(values)
    return {
        "split_name": split_name,
        "task_count": task_count,
        "average_pass_rate": sum(values) / task_count,
        "full_pass_count": sum(value == 1.0 for value in values),
        "full_pass_rate": sum(value == 1.0 for value in values) / task_count,
        "task_scores": task_scores,
    }


def program_payload() -> dict[str, Any]:
    return {
        "complete": {
            "traces": [],
            "train": [],
            "demos": [],
            "signature": {
                "instructions": "Optimized instruction",
                "fields": [
                    {
                        "prefix": "Code Stub:",
                        "description": "Partial Python source.",
                    },
                    {
                        "prefix": "Completed Code:",
                        "description": "Complete executable Python source code.",
                    },
                ],
            },
            "lm": None,
        },
        "metadata": {"dependency_versions": {"python": "3.13", "dspy": "3.2.1"}},
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value
