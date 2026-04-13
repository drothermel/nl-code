from enum import StrEnum

from pydantic import BaseModel


class CompressibleDataset(StrEnum):
    HUMAN_EVAL_PLUS = "evalplus/humanevalplus"


class Task(BaseModel):
    dataset: CompressibleDataset
    task_id: str
    entry_point_name: str
    gt_solution: str
