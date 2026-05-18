from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class CodeDataset(StrEnum):
    HUMANEVAL_PLUS = "evalplus/humanevalplus"
    HUMANEVAL_PRO = "CodeEval-Pro/humaneval-pro"
    MBPP_PRO = "CodeEval-Pro/mbpp-pro"
    BIGCODEBENCH_LITE_PRO = "CodeEval-Pro/bigcodebench-lite-pro"
    CLASS_EVAL = "FudanSELab/ClassEval"


class TaskTarget(BaseModel):
    name: str
    kind: Literal["function", "class"] = "function"


class TaskSource(BaseModel):
    kind: Literal["gt_solution"] = "gt_solution"
    code: str
    language: Literal["python"] = "python"


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: CodeDataset
    task_id: str
    target: TaskTarget
    source: TaskSource
    version: Literal["v3"] = "v3"

    def validate_raw_task_version(self, raw_task: object) -> None:
        raw_version = getattr(raw_task, "version", None)
        if raw_version != self.version:
            raise ValueError(
                f"Task version {self.version!r} does not match raw task version {raw_version!r}"
            )
