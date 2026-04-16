# ruff: noqa: E402

from pathlib import Path
from threading import Lock
from typing import cast
import sys

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.docker

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "ui" / "dataset-explorer"),
)

from backend.main import app
from backend.services import datasets as explorer_service
from conftest import fail_on_hf, make_humaneval_row, make_mbpp_pro_row, mock_hf_dataset
from nl_code.datasets.humaneval_dataset import HumanEvalDataset
from nl_code.datasets.humaneval_task import RawHumanEvalTask
from nl_code.datasets.mbpp_pro_dataset import MbppProDataset
from nl_code.datasets.mbpp_pro_task import RawMbppProTask


def _metric_stats_by_key(
    row: explorer_service.DatasetCompareRow,
) -> dict[str, explorer_service.SummaryStats]:
    return {metric.key: metric.stats for metric in row.metrics}


def _ratio_stats_by_key(
    row: explorer_service.DatasetCompareRow,
) -> dict[str, explorer_service.SummaryStats]:
    return {ratio.key: ratio.stats for ratio in row.ratios}


@pytest.fixture
def configured_compare_registry(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    request.getfixturevalue("dataset_cache_dir")
    human_rows = [
        make_humaneval_row(task_id="HumanEval/0"),
        make_humaneval_row(
            task_id="HumanEval/bad",
            canonical_solution="    return a - b\n",
        ),
    ]
    mbpp_rows = [
        make_mbpp_pro_row(id=0),
        make_mbpp_pro_row(
            id=1,
            new_problem=(
                "# Add each pair and return the running totals.\n"
                "def add_pairs_running(pairs: list[tuple[int, int]]) -> list[int]:\n"
            ),
            new_solution=(
                "    total = 0\n"
                "    result = []\n"
                "    for a, b in pairs:\n"
                "        total += a + b\n"
                "        result.append(total)\n"
                "    return result\n"
            ),
            test_code=(
                "assert add_pairs_running([(1, 2), (3, 4)]) == [3, 10]\n"
                "assert add_pairs_running([]) == []\n"
            ),
        ),
    ]
    rows_by_id = {
        HumanEvalDataset().dataset_id.value: human_rows,
        MbppProDataset().dataset_id.value: mbpp_rows,
    }

    monkeypatch.setattr(
        "nl_code.datasets.dataset.load_dataset",
        lambda hf_id, _split=None, **_kwargs: mock_hf_dataset(rows_by_id[hf_id]),
    )

    human_dataset = HumanEvalDataset().load(force_reparse=True)
    mbpp_dataset = MbppProDataset().load(force_reparse=True)
    monkeypatch.setattr("nl_code.datasets.dataset.load_dataset", fail_on_hf)

    custom_registry = {
        "humaneval-plus": explorer_service.DatasetRegistryEntry(
            key="humaneval-plus",
            dataset_id=human_dataset.dataset_id,
            label="HumanEval+",
            family="humaneval",
            dataset_type=HumanEvalDataset,
            split=human_dataset.split,
        ),
        "mbpp-pro": explorer_service.DatasetRegistryEntry(
            key="mbpp-pro",
            dataset_id=mbpp_dataset.dataset_id,
            label="MBPP Pro",
            family="pro",
            dataset_type=MbppProDataset,
            split=mbpp_dataset.split,
        ),
    }
    custom_cache = {
        "humaneval-plus": human_dataset,
        "mbpp-pro": mbpp_dataset,
    }

    monkeypatch.setattr(explorer_service, "DATASET_REGISTRY", custom_registry)
    monkeypatch.setattr(explorer_service, "DATASET_CACHE", custom_cache)
    monkeypatch.setattr(
        explorer_service,
        "DATASET_LOAD_LOCKS",
        {key: Lock() for key in custom_registry},
    )


def test_get_comparison_summarizes_counts_metrics_and_ratios(
    configured_compare_registry: None,
) -> None:
    response = explorer_service.get_comparison()

    assert [row.dataset.key for row in response.datasets] == [
        "humaneval-plus",
        "mbpp-pro",
    ]

    human_row = response.datasets[0]
    assert human_row.counts.raw_sample_count == 1
    assert human_row.counts.task_count == 1
    assert human_row.counts.flawed_count == 1
    assert human_row.flawed_rate == 1.0

    human_metric_stats = _metric_stats_by_key(human_row)
    human_task = explorer_service.DATASET_CACHE["humaneval-plus"].tasks["HumanEval/0"]
    human_raw = cast(
        RawHumanEvalTask,
        explorer_service.DATASET_CACHE["humaneval-plus"].raw_samples["HumanEval/0"],
    )
    assert human_metric_stats["description_length_chars"].median == float(
        len(human_task.description)
    )
    assert human_metric_stats["prompt_length_chars"].median == float(
        len(human_raw.official_prompt)
    )

    mbpp_row = response.datasets[1]
    mbpp_metric_stats = _metric_stats_by_key(mbpp_row)
    mbpp_ratio_stats = _ratio_stats_by_key(mbpp_row)
    mbpp_dataset = explorer_service.DATASET_CACHE["mbpp-pro"]

    derived_lengths = sorted(
        float(len(task.gt_solution)) for task in mbpp_dataset.tasks.values()
    )
    expected_derived_p90 = derived_lengths[0] + 0.9 * (
        derived_lengths[1] - derived_lengths[0]
    )
    assert mbpp_metric_stats["derived_code_length_chars"].count == 2
    assert mbpp_metric_stats["derived_code_length_chars"].p90 == expected_derived_p90

    ratios = sorted(
        float(len(raw.test_code)) / float(len(task.gt_solution))
        for task_id, task in mbpp_dataset.tasks.items()
        for raw in [cast(RawMbppProTask, mbpp_dataset.raw_samples[task_id])]
    )
    expected_ratio_median = (ratios[0] + ratios[1]) / 2
    expected_ratio_p90 = ratios[0] + 0.9 * (ratios[1] - ratios[0])
    assert (
        mbpp_ratio_stats["test_to_derived_code_ratio"].median == expected_ratio_median
    )
    assert mbpp_ratio_stats["test_to_derived_code_ratio"].p90 == expected_ratio_p90

    metric_series = {series.key: series for series in response.metric_series}
    assert [
        dataset.dataset_key
        for dataset in metric_series["derived_code_length_chars"].datasets
    ] == [
        "humaneval-plus",
        "mbpp-pro",
    ]

    landscape_points = {point.dataset_key: point for point in response.landscape_points}
    assert landscape_points["humaneval-plus"].task_count == 1
    assert landscape_points["mbpp-pro"].task_count == 2


def test_compare_endpoint_returns_compare_payload(
    configured_compare_registry: None,
) -> None:
    client = TestClient(app)

    response = client.get("/api/datasets/compare")

    assert response.status_code == 200
    payload = response.json()
    assert [row["dataset"]["key"] for row in payload["datasets"]] == [
        "humaneval-plus",
        "mbpp-pro",
    ]
    assert any(
        series["key"] == "description_length_chars"
        for series in payload["metric_series"]
    )
    assert any(
        series["key"] == "description_to_prompt_ratio"
        for series in payload["ratio_series"]
    )


def test_refresh_endpoint_reloads_cached_dataset(
    configured_compare_registry: None,
) -> None:
    client = TestClient(app)

    stale_dataset = explorer_service.DATASET_CACHE["humaneval-plus"]
    explorer_service.DATASET_CACHE["humaneval-plus"] = HumanEvalDataset()

    response = client.post("/api/datasets/humaneval-plus/refresh")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"]["key"] == "humaneval-plus"
    assert payload["reloaded"] is True

    refreshed_dataset = explorer_service.DATASET_CACHE["humaneval-plus"]
    assert refreshed_dataset is not stale_dataset
    assert len(refreshed_dataset.tasks) == 1
    assert "HumanEval/0" in refreshed_dataset.tasks
