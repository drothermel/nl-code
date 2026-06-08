from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from typer.testing import CliRunner


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_dspy_gepa_agent_bundle.py"


def test_build_agent_bundle_preserves_normalized_and_best_effort_records(
    tmp_path: Path,
) -> None:
    module = load_script_module()
    reports_dir = gepa_reports_dir(tmp_path)

    bundle = module.build_agent_bundle(reports_dir)

    assert bundle.schema_version == "dspy_gepa_agent_bundle_v0"
    assert len(bundle.sessions) == 2
    assert [session.session_id for session in bundle.sessions] == [
        "session_full",
        "session_missing",
    ]

    prompt_by_id = {prompt.id: prompt for prompt in bundle.prompt_variants}
    baseline_id = (
        "session_full:human_eval_dspy_direct_gepa_optimized_20260515T000000Z"
        ":prompt:baseline"
    )
    candidate_id = (
        "session_full:human_eval_dspy_direct_gepa_optimized_20260515T000000Z"
        ":prompt:candidate:000001"
    )
    optimized_id = (
        "session_full:human_eval_dspy_direct_gepa_optimized_20260515T000000Z"
        ":prompt:optimized"
    )
    assert prompt_by_id[baseline_id].prompt_text == BASELINE_PROMPT
    assert prompt_by_id[baseline_id].performance.valset_score == 0.5
    assert prompt_by_id[optimized_id].prompt_text == FINAL_PROMPT
    assert prompt_by_id[candidate_id].prompt_text == FINAL_PROMPT
    assert prompt_by_id[candidate_id].is_final_saved_prompt is True
    assert prompt_by_id[candidate_id].per_valset_task_scores[0].task_id == "HumanEval/1"
    assert prompt_by_id[candidate_id].per_valset_task_scores[0].pass_rate == 1.0

    missing_baseline = (
        "session_missing:human_eval_dspy_direct_gepa_optimized_20260515T000001Z"
        ":prompt:baseline"
    )
    assert prompt_by_id[missing_baseline].status == module.PromptVariantStatus.MISSING
    assert any(
        record.get("prompt_variant_id") == missing_baseline
        for record in bundle.not_present
    )

    evaluations_by_type = {
        evaluation.evaluation_type for evaluation in bundle.evaluations
    }
    assert evaluations_by_type == {
        module.EvaluationType.SPLIT,
        module.EvaluationType.TASK,
        module.EvaluationType.METRIC_CALL,
        module.EvaluationType.CANDIDATE_VALSET_TASK,
    }
    assert any(
        evaluation.prompt_variant_id == baseline_id
        and evaluation.split == "dev"
        and evaluation.average_pass_rate == 0.5
        for evaluation in bundle.evaluations
    )
    assert any(
        evaluation.prompt_variant_id == candidate_id
        and evaluation.task_id == "HumanEval/1"
        and evaluation.pass_rate == 1.0
        for evaluation in bundle.evaluations
    )

    assert bundle.generated_outputs[0].prompt_variant_id == optimized_id
    assert bundle.generated_outputs[0].completed_code == "def g():\n    return 1\n"
    assert bundle.state_summaries[0].string_keys_seen[:2] == [
        "program_candidates",
        "complete",
    ]
    assert bundle.agent_index["sessions_by_generation_type"] == {
        "direct_gepa": ["session_full", "session_missing"]
    }
    assert (
        bundle.agent_index["final_prompt_variant_by_session"]["session_full"]
        == optimized_id
    )
    assert bundle.agent_index["final_saved_prompt_variants_by_session"][
        "session_full"
    ] == [candidate_id, optimized_id]
    assert "HumanEval/1" in bundle.agent_index["evaluations_by_task_id"]


def test_cli_writes_agent_bundle(tmp_path: Path) -> None:
    module = load_script_module()
    reports_dir = gepa_reports_dir(tmp_path)
    output_path = tmp_path / "bundle.json"
    runner = CliRunner()

    result = runner.invoke(module.app, [str(reports_dir), "-o", str(output_path)])

    assert result.exit_code == 0
    data = read_json(output_path)
    assert data["schema_version"] == "dspy_gepa_agent_bundle_v0"
    assert data["bundle_metadata"]["report_count"] == 2
    assert len(data["prompt_variants"]) == 5


def load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_dspy_gepa_agent_bundle",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load build_dspy_gepa_agent_bundle.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gepa_reports_dir(tmp_path: Path) -> Path:
    reports_dir = tmp_path / "parsed_gepa_reports"
    reports_dir.mkdir()
    full_report = reports_dir / "session_full.gepa_report.json"
    missing_report = reports_dir / "session_missing.gepa_report.json"
    write_json(full_report, parsed_report_payload("session_full", run_suffix="000000Z"))
    write_json(
        missing_report,
        parsed_report_payload(
            "session_missing",
            run_suffix="000001Z",
            include_program_prompts=False,
            include_state=False,
            include_candidate=False,
        ),
    )
    write_json(
        reports_dir / "manifest.json",
        {
            "schema_version": "dspy_gepa_session_report_manifest_v0",
            "created_at": "2026-06-08T00:00:00Z",
            "source_root": str(tmp_path),
            "output_dir": str(reports_dir),
            "session_count": 2,
            "gepa_session_count": 2,
            "skipped_session_count": 0,
            "reports": [
                manifest_item("session_full", full_report),
                manifest_item("session_missing", missing_report),
            ],
        },
    )
    return reports_dir


def parsed_report_payload(
    session_id: str,
    *,
    run_suffix: str,
    include_program_prompts: bool = True,
    include_state: bool = True,
    include_candidate: bool = True,
) -> dict[str, Any]:
    run_id = f"human_eval_dspy_direct_gepa_optimized_20260515T{run_suffix}"
    run_log = f"/tmp/{session_id}/raw/logs/dspy_optimized/{run_id}_run.log"
    programs = [
        program(
            run_id,
            phase="baseline",
            prompt=BASELINE_PROMPT if include_program_prompts else None,
            source_path=f"/tmp/{session_id}/raw/logs/dspy_optimized/{run_id}_summary.json",
        ),
        program(
            run_id,
            phase="optimized",
            prompt=FINAL_PROMPT if include_program_prompts else None,
            source_path=f"/tmp/{session_id}/raw/logs/dspy_optimized/{run_id}.json",
        ),
    ]
    if include_candidate:
        programs.append(
            program(
                run_id,
                phase="candidate",
                prompt=FINAL_PROMPT,
                source_path=run_log,
                candidate_index=1,
            )
        )

    optimizer_iterations = []
    if include_candidate:
        optimizer_iterations.append(
            {
                "id": f"{run_id}:iteration:000001",
                "run_id": run_id,
                "iteration": 1,
                "selected_program_index": 0,
                "selected_score": 0.5,
                "proposed_predictor": "complete",
                "proposal_text": "Rejected short proposal",
                "proposal_char_count": 23,
                "new_subsample_score": 1.0,
                "old_subsample_score": 0.5,
                "subsample_comparison": "better",
                "subsample_action": "full_eval_and_add",
                "valset_score": 1.0,
                "valset_coverage": "1 / 1",
                "individual_valset_scores": {"0": 1.0},
                "pareto_front_scores": {"0": 1.0},
                "pareto_front_programs": {"0": [0, 1]},
                "pareto_aggregate_score": 1.0,
                "best_program_index": 1,
                "best_score": 1.0,
                "linear_pareto_program_index": 1,
                "new_candidate_index": 1,
                "found_better_score": 1.0,
                "skip_messages": [],
                "source_lines": [12, 13],
                "confidence": "best_effort",
            }
        )

    return {
        "schema_version": "dspy_gepa_session_report_v0",
        "created_at": "2026-06-08T00:00:00Z",
        "session": {
            "session_id": session_id,
            "session_dir": f"/tmp/{session_id}",
            "session_kind": "optimization",
        },
        "optimizer_runs": [
            {
                "id": run_id,
                "source_file": f"/tmp/{session_id}/raw/logs/dspy_optimized/{run_id}_summary.json",
                "source_relative_path": f"raw/logs/dspy_optimized/{run_id}_summary.json",
                "timestamp": "2026-05-15T00:00:05Z",
                "generation_type": "direct_gepa",
                "optimization_target": None,
                "model": "openrouter/test-model",
                "llm_config_id": "test-model",
                "reasoning_config": None,
                "auto": None,
                "max_metric_calls": 3,
                "num_threads": 1,
                "seed": 42,
                "split_task_ids": {
                    "train": ["HumanEval/0"],
                    "dev": ["HumanEval/1"],
                    "eval": [],
                },
                "artifact_paths": {"run_log_path": run_log},
                "final_program_id": f"{run_id}:program:optimized",
            }
        ],
        "programs": programs,
        "split_evaluations": [
            split_evaluation(run_id, "baseline", 0.5),
            split_evaluation(run_id, "optimized", 1.0),
        ],
        "task_scores": [
            task_score(run_id, "baseline", 0.5),
            task_score(run_id, "optimized", 1.0),
        ],
        "metric_calls": [
            {
                "id": f"{run_id}:metric_call:000001",
                "run_id": run_id,
                "kind": "metric_call",
                "timestamp": "2026-05-15T00:00:01Z",
                "label": "direct/gepa-eval",
                "metric_call": 1,
                "phase": "baseline",
                "split": "dev",
                "iteration": None,
                "program_id": f"{run_id}:program:baseline",
                "predictor": None,
                "task_id": "HumanEval/1",
                "pass_rate": 0.5,
                "error": None,
                "source": source_ref(f"/tmp/{session_id}/{run_id}_events.jsonl", 1),
                "confidence": "definitely_extractable",
            }
        ],
        "generated_outputs": [
            {
                "id": f"{run_id}:generated_output:000000",
                "run_id": run_id,
                "program_id": f"{run_id}:program:optimized",
                "task_id": "HumanEval/1",
                "valset_index": 0,
                "iteration": 0,
                "program_index": 0,
                "output_fields": {"completed_code": "def g():\n    return 1\n"},
                "completed_code": "def g():\n    return 1\n",
                "source": source_ref(f"/tmp/{session_id}/task_0/iter_0_prog_0.json", 1),
                "confidence": "definitely_extractable",
            }
        ],
        "optimizer_iterations": optimizer_iterations,
        "best_effort": {
            "optimizer_iterations": optimizer_iterations,
            "candidate_rankings": [],
            "candidate_valset_scores": [],
            "candidate_proposals": [],
            "state_key_inventory": [],
        },
        "state_files": state_files(session_id) if include_state else [],
        "aggregates": {
            "run_count": 1,
            "program_count": len(programs),
            "split_evaluation_count": 2,
            "task_score_count": 2,
            "metric_call_count": 1,
            "generated_output_count": 1,
            "optimizer_iteration_count": len(optimizer_iterations),
            "state_file_count": 1 if include_state else 0,
        },
        "parse_notes": [
            "not_present: historical GEPA artifacts before process start are unavailable"
        ],
    }


BASELINE_PROMPT = (
    "Complete the supplied HumanEval function using a direct, readable Python "
    "implementation."
)
FINAL_PROMPT = (
    "Complete the supplied HumanEval function, preserving the given signature, "
    "handling edge cases, and returning only valid Python code."
)


def manifest_item(session_id: str, report_path: Path) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "session_dir": f"/tmp/{session_id}",
        "report_file": str(report_path),
        "optimizer_run_count": 1,
        "program_count": 2,
        "metric_call_count": 1,
        "generated_output_count": 1,
        "generation_types": ["direct_gepa"],
    }


def program(
    run_id: str,
    *,
    phase: str,
    prompt: str | None,
    source_path: str,
    candidate_index: int | None = None,
) -> dict[str, Any]:
    suffix = phase
    if candidate_index is not None:
        suffix = f"candidate:{candidate_index:06d}"
    return {
        "id": f"{run_id}:program:{suffix}",
        "run_id": run_id,
        "phase": phase,
        "candidate_index": candidate_index,
        "predictor_name": "complete",
        "parent_program_id": None,
        "source": source_ref(source_path, 1),
        "signature_instructions": prompt,
        "signature_fields": [],
        "demos": [],
        "train": [],
        "traces": [],
        "lm": None,
        "dependency_versions": {},
        "confidence": "definitely_extractable" if prompt is not None else "not_present",
    }


def split_evaluation(run_id: str, phase: str, score: float) -> dict[str, Any]:
    return {
        "id": f"{run_id}:split:{phase}:dev",
        "run_id": run_id,
        "phase": phase,
        "split": "dev",
        "label": f"direct GEPA {phase}/dev",
        "program_id": f"{run_id}:program:{phase}",
        "task_count": 1,
        "average_pass_rate": score,
        "full_pass_count": 1 if score == 1.0 else 0,
        "full_pass_rate": 1.0 if score == 1.0 else 0.0,
        "source": source_ref(f"/tmp/{run_id}_summary.json", 1),
        "confidence": "definitely_extractable",
    }


def task_score(run_id: str, phase: str, score: float) -> dict[str, Any]:
    return {
        "id": f"{run_id}:task_score:{phase}:dev:000000",
        "run_id": run_id,
        "evaluation_id": f"{run_id}:split:{phase}:dev",
        "phase": phase,
        "split": "dev",
        "task_id": "HumanEval/1",
        "program_id": f"{run_id}:program:{phase}",
        "valset_index": 0,
        "pass_rate": score,
        "error": None,
        "source": source_ref(f"/tmp/{run_id}_summary.json", 1),
        "confidence": "definitely_extractable",
    }


def state_files(session_id: str) -> list[dict[str, Any]]:
    return [
        {
            "source_file": f"/tmp/{session_id}/raw/logs/dspy_optimized/gepa_state.bin",
            "source_relative_path": "raw/logs/dspy_optimized/gepa_state.bin",
            "size_bytes": 123,
            "sha256": "abc123",
            "decoded": False,
            "pickle_protocol": 4,
            "opcode_counts": {},
            "unsafe_opcode_names": ["STACK_GLOBAL"],
            "string_keys_seen": [
                "program_candidates",
                "complete",
                BASELINE_PROMPT,
                FINAL_PROMPT,
                "prog_candidate_val_subscores",
            ],
            "parse_error": None,
        }
    ]


def source_ref(source_file: str, line_number: int) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "source_relative_path": source_file,
        "line_number": line_number,
        "json_path": None,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, Any], value)
