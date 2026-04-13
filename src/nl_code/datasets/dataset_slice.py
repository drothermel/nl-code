import random

from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field
from pydantic import model_validator

from nl_code.datasets.dataset import Dataset
from nl_code.datasets.task import Task


class DatasetSlice(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Dataset
    ids: list[str] = Field(default_factory=list)
    limit: int | None = None
    shuffle: bool = False
    seed: int | None = None
    raw_source_field: str | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be >= 1")
        if self.seed is not None and not self.shuffle:
            raise ValueError("seed requires shuffle=True")
        return self

    def resolve_tasks(self) -> dict[str, Task]:
        task_ids = self._resolve_task_ids()
        return {task_id: self.dataset.tasks[task_id] for task_id in task_ids}

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

    def _resolve_task_ids(self) -> list[str]:
        if not self.ids:
            task_ids = list(self.dataset.tasks)
        else:
            missing = set(self.ids) - set(self.dataset.tasks)
            if missing:
                raise ValueError(f"Task IDs not found in dataset: {missing}")
            task_ids = list(self.ids)

        if self.shuffle:
            random.Random(self.seed).shuffle(task_ids)
        if self.limit is not None:
            return task_ids[: self.limit]
        return task_ids
