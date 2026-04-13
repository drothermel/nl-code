import pytest

from nl_code.datasets.dataset import FlawedSample
from nl_code.datasets.humaneval_pro_dataset import HumanEvalProDataset

from conftest import make_humaneval_pro_row, mock_hf_dataset


class TestHumanEvalProDataset:
    def test_load_valid_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_humaneval_pro_row(id=0)]
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset(rows),
        )
        ds = HumanEvalProDataset()
        ds.load()

        assert len(ds.raw_samples) == 1
        assert "HumanEvalPro/0" in ds.raw_samples
        assert len(ds.flawed_raw_samples) == 0
        assert len(ds.tasks) == 1
        assert "HumanEvalPro/0" in ds.tasks

    def test_task_has_correct_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_humaneval_pro_row(id=0)]
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset(rows),
        )
        ds = HumanEvalProDataset()
        ds.load()

        task = ds.tasks["HumanEvalPro/0"]
        assert task.entry_point_name == "add_pairs"
        assert "list of pairs" in task.description
        assert '"""' not in task.gt_solution

    def test_flawed_rows_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_row = make_humaneval_pro_row(id=99, new_solution="    return []\n")
        good_row = make_humaneval_pro_row(id=0)
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset([bad_row, good_row]),
        )
        ds = HumanEvalProDataset()
        ds.load()

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "HumanEvalPro/99" in ds.flawed_raw_samples
        assert isinstance(ds.flawed_raw_samples["HumanEvalPro/99"], FlawedSample)

    def test_uses_train_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ds = HumanEvalProDataset()
        assert ds.split == "train"
