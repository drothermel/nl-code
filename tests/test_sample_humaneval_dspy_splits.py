from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "sample_humaneval_dspy_splits.py"


def test_load_legacy_task_results(tmp_path: Path) -> None:
    module = load_script_module()
    log_path = tmp_path / "legacy.json"
    log_path.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "dataset_index": 0,
                        "output": {
                            "task_id": "HumanEval/0",
                            "pass_rate": 1.0,
                        },
                    },
                    {
                        "dataset_index": 1,
                        "output": {
                            "task_id": "HumanEval/1",
                            "pass_rate": 0.5,
                        },
                    },
                ]
            }
        )
    )

    results = module.load_task_results(log_path)

    assert set(results) == {"HumanEval/0", "HumanEval/1"}
    assert results["HumanEval/0"].failed is False
    assert results["HumanEval/1"].failed is True


def test_load_package_task_results_filters_generation_type(tmp_path: Path) -> None:
    module = load_script_module()
    log_path = tmp_path / "package.json"
    log_path.write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "generation_type": "direct",
                        "dataset_index": 0,
                        "task_id": "HumanEval/0",
                        "test_pass_rate": 1.0,
                    },
                    {
                        "generation_type": "encdec",
                        "dataset_index": 0,
                        "task_id": "HumanEval/0",
                        "test_pass_rate": 0.0,
                    },
                    {
                        "generation_type": "encdec",
                        "dataset_index": 1,
                        "task_id": "HumanEval/1",
                        "skipped": True,
                    },
                ]
            }
        )
    )

    direct_results = module.load_task_results(log_path, generation_type="direct")
    encdec_results = module.load_task_results(log_path, generation_type="encdec")

    assert set(direct_results) == {"HumanEval/0"}
    assert direct_results["HumanEval/0"].failed is False
    assert set(encdec_results) == {"HumanEval/0", "HumanEval/1"}
    assert encdec_results["HumanEval/0"].failed is True
    assert encdec_results["HumanEval/1"].failed is True


def test_load_package_task_results_requires_generation_type(tmp_path: Path) -> None:
    module = load_script_module()
    log_path = tmp_path / "package.json"
    log_path.write_text(json.dumps({"attempts": []}))

    with pytest.raises(Exception, match="generation_type is required"):
        module.load_task_results(log_path)


def load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sample_humaneval_dspy_splits", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
