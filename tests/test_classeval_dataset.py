import pytest
from typing import cast

from nl_code.datasets.classeval_dataset import ClassEvalDataset
from nl_code.datasets.classeval_task import RawClassEvalTask
from nl_code.datasets.dataset import FlawedSample

from conftest import make_classeval_row, prime_dataset_cache

pytestmark = pytest.mark.docker


@pytest.mark.usefixtures("dataset_cache_dir")
class TestClassEvalDataset:
    def test_load_valid_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_classeval_row(task_id="ClassEval_0")]
        ds = prime_dataset_cache(ClassEvalDataset(), rows, monkeypatch)

        assert len(ds.raw_samples) == 1
        assert "ClassEval_0" in ds.raw_samples
        assert len(ds.flawed_raw_samples) == 0
        assert len(ds.tasks) == 1
        assert "ClassEval_0" in ds.tasks

    def test_task_has_correct_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_classeval_row(task_id="ClassEval_0")]
        ds = prime_dataset_cache(ClassEvalDataset(), rows, monkeypatch)

        task = ds.tasks["ClassEval_0"]
        raw_task = cast(RawClassEvalTask, ds.raw_samples["ClassEval_0"])
        assert task.entry_point_name == "Calculator"
        assert task.description == "A simple calculator."
        assert "class Calculator" in task.gt_solution
        assert task.gt_solution == raw_task.gt_code

    def test_flawed_rows_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_row = make_classeval_row(
            task_id="ClassEval_99",
            solution_code="class Calculator:\n    def add(self, a, b):\n        return 0\n",
        )
        good_row = make_classeval_row(task_id="ClassEval_0")
        ds = prime_dataset_cache(ClassEvalDataset(), [bad_row, good_row], monkeypatch)

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "ClassEval_99" in ds.flawed_raw_samples
        flawed = ds.flawed_raw_samples["ClassEval_99"]
        assert isinstance(flawed, FlawedSample)
        assert flawed.error.startswith("dataset_failure:")

    def test_uses_test_split(self) -> None:
        ds = ClassEvalDataset()
        assert ds.split == "test"

    def test_known_dataset_issue_is_tracked_as_dataset_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = prime_dataset_cache(
            ClassEvalDataset(),
            [make_classeval_row(task_id="ClassEval_58")],
            monkeypatch,
        )

        assert "ClassEval_58" in ds.flawed_raw_samples
        assert ds.flawed_raw_samples["ClassEval_58"].error.startswith(
            "dataset_failure:"
        )
