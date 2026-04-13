from typing import Any

from pydantic import BaseModel

from nl_code.datasets.bigcodebench_lite_pro_task import RawBigCodeBenchLiteProTask
from nl_code.datasets.dataset import Dataset
from nl_code.datasets.task import CodeDataset, Task


def _parse_task_number(raw_id: str) -> str:
    """Extract the numeric part from a BigCodeBench id like 'BigCodeBench/23'."""
    parts = str(raw_id).split("/")
    return parts[-1]


class BigCodeBenchLiteProDataset(Dataset):
    dataset_id: CodeDataset = CodeDataset.BIGCODEBENCH_LITE_PRO
    split: str = "train"

    def _parse_row(self, row: dict[str, Any]) -> RawBigCodeBenchLiteProTask:
        task_num = _parse_task_number(row["id"])
        row["task_id"] = f"BigCodeBenchLitePro/{task_num}"
        return RawBigCodeBenchLiteProTask.model_validate(row)

    def _extract_task_id(self, row: dict[str, Any]) -> str:
        task_num = _parse_task_number(row["id"])
        return f"BigCodeBenchLitePro/{task_num}"

    def _to_task(self, task_id: str, raw: BaseModel) -> Task:
        assert isinstance(raw, RawBigCodeBenchLiteProTask)
        return Task(
            dataset=self.dataset_id,
            task_id=task_id,
            entry_point_name=raw.new_entry_point,
            description=raw.new_description,
            gt_solution=raw.gt_solution_without_comments,
        )
