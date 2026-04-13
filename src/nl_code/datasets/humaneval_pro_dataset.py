from typing import Any

from pydantic import BaseModel

from nl_code.datasets.dataset import Dataset
from nl_code.datasets.humaneval_pro_task import RawHumanEvalProTask
from nl_code.datasets.task import CodeDataset, Task


class HumanEvalProDataset(Dataset):
    dataset_id: CodeDataset = CodeDataset.HUMANEVAL_PRO
    split: str = "train"

    def _parse_row(self, row: dict[str, Any]) -> RawHumanEvalProTask:
        row["task_id"] = f"HumanEvalPro/{row['id']}"
        return RawHumanEvalProTask.model_validate(row)

    def _extract_task_id(self, row: dict[str, Any]) -> str:
        return f"HumanEvalPro/{row['id']}"

    def _to_task(self, task_id: str, raw: BaseModel) -> Task:
        assert isinstance(raw, RawHumanEvalProTask)
        return Task(
            dataset=self.dataset_id,
            task_id=task_id,
            entry_point_name=raw.new_entry_point,
            description=raw.new_description,
            gt_solution=raw.gt_solution_without_comments,
        )
