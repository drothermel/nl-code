import pytest

from nl_code.datasets.dataset_slice import DatasetSlice
from nl_code.datasets.humaneval_dataset import HumanEvalDataset


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

    def test_resolve_missing_raises(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(dataset=loaded_dataset, ids=["HumanEval/999"])
        with pytest.raises(ValueError, match="not found"):
            sl.resolve_tasks()

    def test_get_source_code_with_field(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(
            dataset=loaded_dataset,
            raw_source_field="gt_solution_without_comments",
        )
        code = sl.get_source_code("HumanEval/0")
        assert "add" in code
        assert '"""' not in code

    def test_get_source_code_none_field(self, loaded_dataset: HumanEvalDataset) -> None:
        sl = DatasetSlice(dataset=loaded_dataset, raw_source_field=None)
        code = sl.get_source_code("HumanEval/0")
        assert "add" in code
