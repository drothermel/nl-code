from enum import StrEnum

from pydantic import BaseModel


class CodeDataset(StrEnum):
    HUMANEVAL_PLUS = "evalplus/humanevalplus"
    HUMANEVAL_PRO = "CodeEval-Pro/humaneval-pro"
    MBPP_PRO = "CodeEval-Pro/mbpp-pro"
    BIGCODEBENCH_LITE_PRO = "CodeEval-Pro/bigcodebench-lite-pro"
    CLASS_EVAL = "FudanSELab/ClassEval"


class Task(BaseModel):
    dataset: CodeDataset
    task_id: str
    entry_point_name: str
    description: str
    gt_solution: str
