from enum import StrEnum
from typing import Literal

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
    version: Literal["v1", "v2"] = "v2"

    def validate_raw_task_version(self, raw_task: object) -> None:
        raw_version = getattr(raw_task, "version", None)
        if raw_version != self.version:
            raise ValueError(
                f"Task version {self.version!r} does not match raw task version {raw_version!r}"
            )
