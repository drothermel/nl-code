import pytest

from nl_code.datasets.bigcodebench_lite_pro_dataset import BigCodeBenchLiteProDataset
from nl_code.datasets.dataset import FlawedSample

from conftest import make_bigcodebench_lite_pro_row, mock_hf_dataset


class TestBigCodeBenchLiteProDataset:
    def test_load_valid_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_bigcodebench_lite_pro_row(id="BigCodeBench/23")]
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset(rows),
        )
        ds = BigCodeBenchLiteProDataset()
        ds.load()

        assert len(ds.raw_samples) == 1
        assert "BigCodeBenchLitePro/23" in ds.raw_samples
        assert len(ds.flawed_raw_samples) == 0
        assert len(ds.tasks) == 1
        assert "BigCodeBenchLitePro/23" in ds.tasks

    def test_task_has_correct_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [make_bigcodebench_lite_pro_row(id="BigCodeBench/23")]
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset(rows),
        )
        ds = BigCodeBenchLiteProDataset()
        ds.load()

        task = ds.tasks["BigCodeBenchLitePro/23"]
        assert task.entry_point_name == "multiply_pairs"
        assert "list of pairs" in task.description
        assert '"""' not in task.gt_solution

    def test_flawed_rows_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_row = make_bigcodebench_lite_pro_row(
            id="BigCodeBench/99", new_solution="    return []\n"
        )
        good_row = make_bigcodebench_lite_pro_row(id="BigCodeBench/23")
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset([bad_row, good_row]),
        )
        ds = BigCodeBenchLiteProDataset()
        ds.load()

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "BigCodeBenchLitePro/99" in ds.flawed_raw_samples
        assert isinstance(ds.flawed_raw_samples["BigCodeBenchLitePro/99"], FlawedSample)

    def test_uses_train_split(self) -> None:
        ds = BigCodeBenchLiteProDataset()
        assert ds.split == "train"

    def test_parses_task_number_from_string_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [make_bigcodebench_lite_pro_row(id="BigCodeBench/456")]
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset(rows),
        )
        ds = BigCodeBenchLiteProDataset()
        ds.load()

        assert "BigCodeBenchLitePro/456" in ds.tasks
