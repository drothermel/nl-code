import pytest

from nl_code.datasets.dataset import FlawedSample
from nl_code.datasets.humaneval_pro_dataset import HumanEvalProDataset

from conftest import make_humaneval_pro_row, prime_dataset_cache


@pytest.mark.usefixtures("dataset_cache_dir")
class TestHumanEvalProDataset:
    def test_load_valid_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_humaneval_pro_row(id=0)]
        ds = prime_dataset_cache(HumanEvalProDataset(), rows, monkeypatch)

        assert len(ds.raw_samples) == 1
        assert "HumanEvalPro/0" in ds.raw_samples
        assert len(ds.flawed_raw_samples) == 0
        assert len(ds.tasks) == 1
        assert "HumanEvalPro/0" in ds.tasks

    def test_task_has_correct_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_humaneval_pro_row(id=0)]
        ds = prime_dataset_cache(HumanEvalProDataset(), rows, monkeypatch)

        task = ds.tasks["HumanEvalPro/0"]
        assert task.entry_point_name == "add_pairs"
        assert "list of pairs" in task.description
        assert '"""' not in task.gt_solution

    def test_flawed_rows_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_row = make_humaneval_pro_row(id=99, new_solution="    return []\n")
        good_row = make_humaneval_pro_row(id=0)
        ds = prime_dataset_cache(
            HumanEvalProDataset(), [bad_row, good_row], monkeypatch
        )

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "HumanEvalPro/99" in ds.flawed_raw_samples
        flawed = ds.flawed_raw_samples["HumanEvalPro/99"]
        assert isinstance(flawed, FlawedSample)
        assert flawed.error.startswith("dataset_failure:")

    def test_uses_train_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ds = HumanEvalProDataset()
        assert ds.split == "train"
