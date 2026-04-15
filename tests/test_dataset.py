import pytest
from pydantic import BaseModel
from typing import ClassVar

from nl_code.datasets.dataset import Dataset, DatasetCacheMissError, FlawedSample
from nl_code.datasets.task import CodeDataset, Task

from conftest import fail_on_hf, mock_hf_dataset, prime_dataset_cache


class _DummyRaw(BaseModel):
    task_id: str
    value: str


class _DummyDataset(Dataset):
    dataset_key: ClassVar[str] = "dummy"
    raw_model_type: ClassVar[type[BaseModel]] = _DummyRaw
    source_revision: ClassVar[str] = "dummy-revision"
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


@pytest.mark.usefixtures("dataset_cache_dir")
class TestDatasetBase:
    def test_load_populates_all_dicts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [
            {"task_id": "t/0", "value": "x = 1"},
            {"task_id": "t/1", "value": "y = 2"},
        ]
        ds = prime_dataset_cache(_DummyDataset(), rows, monkeypatch)
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
        ds = prime_dataset_cache(_DummyDataset(), rows, monkeypatch)

        assert len(ds.raw_samples) == 1
        assert len(ds.flawed_raw_samples) == 1
        assert "t/bad" in ds.flawed_raw_samples
        assert isinstance(ds.flawed_raw_samples["t/bad"], FlawedSample)

    def test_load_returns_self(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ds = prime_dataset_cache(
            _DummyDataset(), [{"task_id": "t/0", "value": "x"}], monkeypatch
        )
        assert ds.load() is ds

    def test_get_task_at_index_uses_load_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {"task_id": "t/0", "value": "x = 1"},
            {"task_id": "t/1", "value": "y = 2"},
        ]
        ds = prime_dataset_cache(_DummyDataset(), rows, monkeypatch)

        assert ds.get_task_at_index(0).task_id == "t/0"
        assert ds.get_task_at_index(1).task_id == "t/1"
        assert ds.get_task_at_index(-1).task_id == "t/1"

    def test_get_raw_sample_at_index_uses_load_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {"task_id": "t/0", "value": "x = 1"},
            {"task_id": "t/1", "value": "y = 2"},
        ]
        ds = prime_dataset_cache(_DummyDataset(), rows, monkeypatch)

        first = ds.get_raw_sample_at_index(0)
        last = ds.get_raw_sample_at_index(-1)

        assert isinstance(first, _DummyRaw)
        assert first.task_id == "t/0"
        assert first.value == "x = 1"
        assert isinstance(last, _DummyRaw)
        assert last.task_id == "t/1"
        assert last.value == "y = 2"

    def test_index_access_skips_flawed_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {"task_id": "t/0", "value": "ok"},
            {"task_id": "t/bad"},
            {"task_id": "t/2", "value": "still ok"},
        ]
        ds = prime_dataset_cache(_DummyDataset(), rows, monkeypatch)

        assert ds.get_task_at_index(0).task_id == "t/0"
        assert ds.get_task_at_index(1).task_id == "t/2"
        raw = ds.get_raw_sample_at_index(1)
        assert isinstance(raw, _DummyRaw)
        assert raw.task_id == "t/2"

    def test_index_access_raises_for_out_of_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = prime_dataset_cache(
            _DummyDataset(), [{"task_id": "t/0", "value": "x"}], monkeypatch
        )

        with pytest.raises(IndexError, match="task index 1 out of range"):
            ds.get_task_at_index(1)
        with pytest.raises(IndexError, match="raw sample index -2 out of range"):
            ds.get_raw_sample_at_index(-2)

    def test_index_access_matches_rebuilt_and_cached_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            {"task_id": "t/0", "value": "x = 1"},
            {"task_id": "t/1", "value": "y = 2"},
        ]
        monkeypatch.setattr(
            "nl_code.datasets.dataset.load_dataset",
            lambda *_a, **_kw: mock_hf_dataset(rows),
        )
        rebuilt = _DummyDataset().load(force_reparse=True)
        monkeypatch.setattr("nl_code.datasets.dataset.load_dataset", fail_on_hf)
        cached = _DummyDataset().load()

        assert rebuilt.get_task_at_index(1).task_id == cached.get_task_at_index(
            1
        ).task_id
        rebuilt_raw = rebuilt.get_raw_sample_at_index(-1)
        cached_raw = cached.get_raw_sample_at_index(-1)
        assert isinstance(rebuilt_raw, _DummyRaw)
        assert isinstance(cached_raw, _DummyRaw)
        assert rebuilt_raw.task_id == cached_raw.task_id
        assert rebuilt_raw.value == cached_raw.value

    def test_hf_id_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[str] = []

        def mock_load(*args: object, **kwargs: object) -> object:
            captured.append(str(args[0]))
            return mock_hf_dataset([{"task_id": "t/0", "value": "x"}])

        monkeypatch.setattr("nl_code.datasets.dataset.load_dataset", mock_load)
        ds = _DummyDataset()
        ds.load(hf_id="custom/dataset", force_reparse=True)
        assert captured[0] == "custom/dataset"

    def test_task_id_fallback_on_bad_extract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [{"value": "x"}]  # no task_id key
        ds = prime_dataset_cache(_DummyDataset(), rows, monkeypatch)

        assert len(ds.flawed_raw_samples) == 1
        assert "row-1" in ds.flawed_raw_samples

    def test_load_without_cache_raises_actionable_error(self) -> None:
        with pytest.raises(DatasetCacheMissError, match="cache_cli rebuild dummy"):
            _DummyDataset().load()
