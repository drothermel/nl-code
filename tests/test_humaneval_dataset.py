import pytest

from nl_code.datasets.dataset import FlawedSample
from nl_code.datasets.humaneval_dataset import HumanEvalDataset

from conftest import make_humaneval_row, mock_hf_dataset


class TestHumanEvalDataset:
    def test_load_valid_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_humaneval_row(task_id="HumanEval/0")]
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset(rows),
        )
        ds = HumanEvalDataset()
        ds.load()

        assert len(ds.raw_samples) == 1
        assert "HumanEval/0" in ds.raw_samples
        assert len(ds.flawed_raw_samples) == 0
        assert len(ds.tasks) == 1
        assert "HumanEval/0" in ds.tasks

    def test_flawed_rows_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_row = make_humaneval_row(
            task_id="HumanEval/bad",
            canonical_solution="    return a - b\n",
        )
        good_row = make_humaneval_row(task_id="HumanEval/0")
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset([bad_row, good_row]),
        )
        ds = HumanEvalDataset()
        ds.load()

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "HumanEval/bad" in ds.flawed_raw_samples
        flawed = ds.flawed_raw_samples["HumanEval/bad"]
        assert isinstance(flawed, FlawedSample)
        assert flawed.error

    def test_derived_tasks_use_stripped_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset([make_humaneval_row()]),
        )
        ds = HumanEvalDataset()
        ds.load()
        task = ds.tasks["HumanEval/0"]
        assert '"""' not in task.gt_solution

    def test_derived_tasks_have_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset([make_humaneval_row()]),
        )
        ds = HumanEvalDataset()
        ds.load()
        task = ds.tasks["HumanEval/0"]
        assert task.description == "Add two integers and return the result."
