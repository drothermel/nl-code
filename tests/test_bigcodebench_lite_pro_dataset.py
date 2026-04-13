import pytest

from nl_code.datasets.bigcodebench_lite_pro_dataset import BigCodeBenchLiteProDataset
from nl_code.datasets.dataset import FlawedSample

from conftest import make_bigcodebench_lite_pro_row, prime_dataset_cache


class TestBigCodeBenchLiteProDataset:
    def test_load_valid_rows(
        self, monkeypatch: pytest.MonkeyPatch, dataset_cache_dir: object
    ) -> None:
        rows = [make_bigcodebench_lite_pro_row(id="BigCodeBench/23")]
        ds = prime_dataset_cache(BigCodeBenchLiteProDataset(), rows, monkeypatch)

        assert len(ds.raw_samples) == 1
        assert "BigCodeBenchLitePro/23" in ds.raw_samples
        assert len(ds.flawed_raw_samples) == 0
        assert len(ds.tasks) == 1
        assert "BigCodeBenchLitePro/23" in ds.tasks

    def test_task_has_correct_fields(
        self, monkeypatch: pytest.MonkeyPatch, dataset_cache_dir: object
    ) -> None:
        rows = [make_bigcodebench_lite_pro_row(id="BigCodeBench/23")]
        ds = prime_dataset_cache(BigCodeBenchLiteProDataset(), rows, monkeypatch)

        task = ds.tasks["BigCodeBenchLitePro/23"]
        assert task.entry_point_name == "multiply_pairs"
        assert "list of pairs" in task.description
        assert '"""' not in task.gt_solution

    def test_flawed_rows_tracked(
        self, monkeypatch: pytest.MonkeyPatch, dataset_cache_dir: object
    ) -> None:
        bad_row = make_bigcodebench_lite_pro_row(
            id="BigCodeBench/99", new_solution="    return []\n"
        )
        good_row = make_bigcodebench_lite_pro_row(id="BigCodeBench/23")
        ds = prime_dataset_cache(
            BigCodeBenchLiteProDataset(), [bad_row, good_row], monkeypatch
        )

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "BigCodeBenchLitePro/99" in ds.flawed_raw_samples
        assert isinstance(ds.flawed_raw_samples["BigCodeBenchLitePro/99"], FlawedSample)

    def test_uses_train_split(self) -> None:
        ds = BigCodeBenchLiteProDataset()
        assert ds.split == "train"

    def test_parses_task_number_from_string_id(
        self, monkeypatch: pytest.MonkeyPatch, dataset_cache_dir: object
    ) -> None:
        rows = [make_bigcodebench_lite_pro_row(id="BigCodeBench/456")]
        ds = prime_dataset_cache(BigCodeBenchLiteProDataset(), rows, monkeypatch)

        assert "BigCodeBenchLitePro/456" in ds.tasks
