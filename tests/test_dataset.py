import pytest
from pydantic import BaseModel

from nl_code.datasets.dataset import Dataset, FlawedSample
from nl_code.datasets.task import CodeDataset, Task

from conftest import mock_hf_dataset


class _DummyRaw(BaseModel):
    task_id: str
    value: str


class _DummyDataset(Dataset):
    dataset_id: CodeDataset = CodeDataset.HUMANEVAL_PLUS

    def _parse_row(self, row: dict) -> _DummyRaw:
        return _DummyRaw(task_id=row["task_id"], value=row["value"])

    def _extract_task_id(self, row: dict) -> str:
        return str(row["task_id"])

    def _to_task(self, task_id: str, raw: BaseModel) -> Task:
        assert isinstance(raw, _DummyRaw)
        return Task(
            dataset=self.dataset_id,
            task_id=task_id,
            entry_point_name="main",
            description="dummy",
            gt_solution=raw.value,
        )


class TestDatasetBase:
    def test_load_populates_all_dicts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [
            {"task_id": "t/0", "value": "x = 1"},
            {"task_id": "t/1", "value": "y = 2"},
        ]
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset(rows),
        )
        ds = _DummyDataset()
        result = ds.load()

        assert result is ds
        assert len(ds.raw_samples) == 2
        assert len(ds.tasks) == 2
        assert len(ds.flawed_raw_samples) == 0

    def test_flawed_rows_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [
            {"task_id": "t/0", "value": "ok"},
            {"task_id": "t/bad"},  # missing "value" field
        ]
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset(rows),
        )
        ds = _DummyDataset()
        ds.load()

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "t/bad" in ds.flawed_raw_samples
        assert isinstance(ds.flawed_raw_samples["t/bad"], FlawedSample)

    def test_load_returns_self(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset([{"task_id": "t/0", "value": "x"}]),
        )
        ds = _DummyDataset()
        assert ds.load() is ds

    def test_hf_id_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[str] = []

        def mock_load(*args: object, **kwargs: object) -> object:
            captured.append(str(args[0]))
            return mock_hf_dataset([{"task_id": "t/0", "value": "x"}])

        monkeypatch.setattr("nl_code.datasets.dataset.load_dataset", mock_load)
        ds = _DummyDataset()
        ds.load(hf_id="custom/dataset")
        assert captured[0] == "custom/dataset"

    def test_task_id_fallback_on_bad_extract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [{"value": "x"}]  # no task_id key
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *a, **kw: mock_hf_dataset(rows),
        )
        ds = _DummyDataset()
        ds.load()

        assert len(ds.flawed_raw_samples) == 1
        assert "row-1" in ds.flawed_raw_samples
