import pytest

from nl_code.datasets.dataset_slice import DatasetSlice
from nl_code.datasets.humaneval_dataset import HumanEvalDataset

pytestmark = pytest.mark.docker


class TestDatasetSlice:
    def test_resolve_all_tasks(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(dataset=loaded_dataset)
        tasks = sl.resolve_tasks()
        assert len(tasks) == 2

    def test_resolve_filtered_tasks(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(dataset=loaded_dataset, ids=["HumanEval/0"])
        tasks = sl.resolve_tasks()
        assert len(tasks) == 1
        assert "HumanEval/0" in tasks

    def test_resolve_limited_tasks_from_dataset_order(
        self, loaded_dataset: HumanEvalDataset
    ) -> None:
        sl = DatasetSlice(dataset=loaded_dataset, limit=1)
        tasks = sl.resolve_tasks()
        assert list(tasks) == ["HumanEval/0"]

    def test_resolve_limited_tasks_from_filtered_ids(
        self, loaded_dataset: HumanEvalDataset
    ) -> None:
        sl = DatasetSlice(
            dataset=loaded_dataset,
            ids=["HumanEval/1", "HumanEval/0"],
            limit=1,
        )
        tasks = sl.resolve_tasks()
        assert list(tasks) == ["HumanEval/1"]

    def test_resolve_limit_larger_than_available_returns_all(
        self, loaded_dataset: HumanEvalDataset
    ) -> None:
        sl = DatasetSlice(dataset=loaded_dataset, limit=10)
        tasks = sl.resolve_tasks()
        assert list(tasks) == ["HumanEval/0", "HumanEval/1"]

    def test_resolve_shuffled_tasks_is_seeded(
        self, loaded_dataset: HumanEvalDataset
    ) -> None:
        sl = DatasetSlice(dataset=loaded_dataset, shuffle=True, seed=7)
        tasks = sl.resolve_tasks()
        assert list(tasks) == ["HumanEval/0", "HumanEval/1"]

    def test_resolve_shuffled_and_limited_tasks_applies_shuffle_before_limit(
        self, loaded_dataset: HumanEvalDataset
    ) -> None:
        sl = DatasetSlice(
            dataset=loaded_dataset,
            ids=["HumanEval/1", "HumanEval/0"],
            shuffle=True,
            seed=1,
            limit=1,
        )
        tasks = sl.resolve_tasks()
        assert list(tasks) == ["HumanEval/0"]

    def test_resolve_missing_raises(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(dataset=loaded_dataset, ids=["HumanEval/999"])
        with pytest.raises(ValueError, match="not found"):
            sl.resolve_tasks()

    def test_resolve_duplicate_ids_raises(
        self, loaded_dataset: HumanEvalDataset
    ) -> None:
        sl = DatasetSlice(
            dataset=loaded_dataset,
            ids=["HumanEval/0", "HumanEval/0"],
        )
        with pytest.raises(ValueError, match="duplicate task ids"):
            sl.resolve_tasks()

    def test_limit_must_be_positive(self, loaded_dataset: HumanEvalDataset) -> None:
        with pytest.raises(ValueError, match="limit must be >= 1"):
            DatasetSlice(dataset=loaded_dataset, limit=0)

    def test_seed_requires_shuffle(self, loaded_dataset: HumanEvalDataset) -> None:
        with pytest.raises(ValueError, match="seed requires shuffle=True"):
            DatasetSlice(dataset=loaded_dataset, seed=7)

    def test_get_source_code_with_field(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(
            dataset=loaded_dataset,
            raw_source_field="gt_solution",
        )
        code = sl.get_source_code("HumanEval/0")
        assert "add" in code
        assert '"""' not in code

    def test_get_source_code_none_field(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(dataset=loaded_dataset, raw_source_field=None)
        code = sl.get_source_code("HumanEval/0")
        assert "add" in code

    def test_get_official_prompt(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(dataset=loaded_dataset)
        prompt = sl.get_official_prompt("HumanEval/0")
        assert prompt.startswith("Read the following function signature and docstring")
        assert "```python\n" in prompt

    def test_get_code_stub(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(dataset=loaded_dataset)
        code_stub = sl.get_code_stub("HumanEval/0")
        assert "def add" in code_stub
        assert '"""' not in code_stub

    def test_get_code_stub_with_comments(
        self, loaded_dataset: HumanEvalDataset
    ) -> None:
        sl = DatasetSlice(dataset=loaded_dataset)
        code_stub = sl.get_code_stub_with_comments("HumanEval/0")
        assert "def add" in code_stub
        assert '"""' in code_stub
