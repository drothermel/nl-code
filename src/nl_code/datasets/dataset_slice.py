from pydantic import BaseModel, ConfigDict, Field

from nl_code.datasets.humaneval_dataset import HumanEvalDataset
from nl_code.datasets.humaneval_task import HumanEvalTask


class DatasetSlice(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: HumanEvalDataset
    ids: list[str] = Field(default_factory=list)
    raw_source_field: str | None = "gt_solution_without_comments"

    def resolve_tasks(self) -> dict[str, HumanEvalTask]:
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
        return getattr(raw, self.raw_source_field)
