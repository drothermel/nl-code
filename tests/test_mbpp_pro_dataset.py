import pytest

from nl_code.datasets.dataset import FlawedSample
from nl_code.datasets.mbpp_pro_dataset import MbppProDataset

from conftest import make_mbpp_pro_row, prime_dataset_cache


class TestMbppProDataset:
    def test_load_valid_rows(
        self, monkeypatch: pytest.MonkeyPatch, dataset_cache_dir: object
    ) -> None:
        rows = [make_mbpp_pro_row(id=0)]
        ds = prime_dataset_cache(MbppProDataset(), rows, monkeypatch)

        assert len(ds.raw_samples) == 1
        assert "MbppPro/0" in ds.raw_samples
        assert len(ds.flawed_raw_samples) == 0
        assert len(ds.tasks) == 1
        assert "MbppPro/0" in ds.tasks

    def test_task_has_correct_fields(
        self, monkeypatch: pytest.MonkeyPatch, dataset_cache_dir: object
    ) -> None:
        rows = [make_mbpp_pro_row(id=0)]
        ds = prime_dataset_cache(MbppProDataset(), rows, monkeypatch)

        task = ds.tasks["MbppPro/0"]
        assert task.entry_point_name == "add_pairs"
        assert "list of pairs" in task.description
        assert '"""' not in task.gt_solution

    def test_flawed_rows_tracked(
        self, monkeypatch: pytest.MonkeyPatch, dataset_cache_dir: object
    ) -> None:
        bad_row = make_mbpp_pro_row(id=99, new_solution="    return []\n")
        good_row = make_mbpp_pro_row(id=0)
        ds = prime_dataset_cache(MbppProDataset(), [bad_row, good_row], monkeypatch)

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "MbppPro/99" in ds.flawed_raw_samples
        assert isinstance(ds.flawed_raw_samples["MbppPro/99"], FlawedSample)

    def test_uses_train_split(self) -> None:
        ds = MbppProDataset()
        assert ds.split == "train"
