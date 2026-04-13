from typing import Any, ClassVar

from pydantic import BaseModel

from nl_code.datasets.classeval_task import RawClassEvalTask
from nl_code.datasets.dataset import Dataset
from nl_code.datasets.task import CodeDataset, Task


class ClassEvalDataset(Dataset):
    dataset_key: ClassVar[str] = "class-eval"
    raw_model_type: ClassVar[type[BaseModel]] = RawClassEvalTask
    source_revision: ClassVar[str] = "fef204b34e221f207f47904ee660bb920d4c5d1d"
    dataset_id: CodeDataset = CodeDataset.CLASS_EVAL

    def _parse_row(self, row: dict[str, Any]) -> RawClassEvalTask:
        return RawClassEvalTask.model_validate(row)

    def _extract_task_id(self, row: dict[str, Any]) -> str:
        return str(row["task_id"])

    def _to_task(self, task_id: str, raw: BaseModel) -> Task:
        assert isinstance(raw, RawClassEvalTask)
        return Task(
            dataset=self.dataset_id,
            task_id=task_id,
            entry_point_name=raw.class_name,
            description=raw.class_description,
            gt_solution=raw.gt_code,
        )
