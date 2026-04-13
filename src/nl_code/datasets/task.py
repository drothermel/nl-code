from enum import StrEnum

from pydantic import BaseModel


class CodeDataset(StrEnum):
    HUMAN_EVAL_PLUS = "evalplus/humanevalplus"
    HUMAN_EVAL_PRO = "CodeEval-Pro/humaneval-pro"


class Task(BaseModel):
    dataset: CodeDataset
    task_id: str
    entry_point_name: str
    description: str
    gt_solution: str
