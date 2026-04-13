from pydantic import BaseModel, ConfigDict, Field

from nl_code.datasets.dataset import Dataset
from nl_code.datasets.task import Task


class DatasetSlice(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Dataset
    ids: list[str] = Field(default_factory=list)
    raw_source_field: str | None = "gt_solution_without_comments"

    def resolve_tasks(self) -> dict[str, Task]:
        if not self.ids:
            return dict(self.dataset.tasks)
        missing = set(self.ids) - set(self.dataset.tasks)
        if missing:
            raise ValueError(f"Task IDs not found in dataset: {missing}")
        return {tid: self.dataset.tasks[tid] for tid in self.ids}

    def get_source_code(self, task_id: str) -> str:
        if self.raw_source_field is None:
            return self.dataset.tasks[task_id].gt_solution
        raw = self.dataset.raw_samples[task_id]
        if not hasattr(raw, self.raw_source_field):
            raise AttributeError(
                f"Task {task_id!r}: raw sample has no field {self.raw_source_field!r}"
            )
        value = getattr(raw, self.raw_source_field)
        if not isinstance(value, str):
            raise TypeError(
                f"Task {task_id!r}: field {self.raw_source_field!r} is {type(value).__name__}, expected str"
            )
        return value
