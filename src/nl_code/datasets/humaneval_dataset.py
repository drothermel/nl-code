from typing import Any

from pydantic import BaseModel

from nl_code.datasets.dataset import Dataset
from nl_code.datasets.humaneval_task import RawHumanEvalTask
from nl_code.datasets.task import CodeDataset, Task


class HumanEvalDataset(Dataset):
    dataset_id: CodeDataset = CodeDataset.HUMANEVAL_PLUS

    def _parse_row(self, row: dict[str, Any]) -> RawHumanEvalTask:
        return RawHumanEvalTask.model_validate(row)

    def _extract_task_id(self, row: dict[str, Any]) -> str:
        return str(row["task_id"])

    def _to_task(self, task_id: str, raw: BaseModel) -> Task:
        assert isinstance(raw, RawHumanEvalTask)
        return Task(
            dataset=self.dataset_id,
            task_id=task_id,
            entry_point_name=raw.entry_point,
            description=raw.prompt_docstring,
            gt_solution=raw.gt_solution_without_comments,
        )
