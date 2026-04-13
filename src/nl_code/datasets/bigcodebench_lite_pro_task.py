import ast

from typing import Self

from pydantic import BaseModel, Field, model_validator

from nl_code.code_parsing import merge_code_components, remove_docstrings_and_comments
from nl_code.datasets.pro_task_helpers import (
    build_gt_solution,
    extract_new_description,
    extract_new_entry_point,
)


class RawBigCodeBenchLiteProTask(BaseModel):
    task_id: str
    raw_problem: str
    raw_solution: str
    new_problem: str
    new_solution: str
    test_code: str
    validated: bool = False

    gt_solution: str = Field(
        default_factory=lambda data: build_gt_solution(
            data.get("raw_problem"),
            data.get("raw_solution"),
            data.get("new_problem"),
            data.get("new_solution"),
        )
    )
    gt_solution_without_comments: str = Field(
        default_factory=lambda data: remove_docstrings_and_comments(
            data.get("gt_solution")
        )
    )
    new_entry_point: str = Field(
        default_factory=lambda data: extract_new_entry_point(
            data.get("new_problem"), data.get("new_solution")
        )
    )
    new_description: str = Field(
        default_factory=lambda data: extract_new_description(data.get("new_problem"))
    )

    @model_validator(mode="after")
    def validate_eval_task(self) -> Self:
        if self.validated:
            return self

        ast.parse(self.gt_solution)
        ast.parse(self.gt_solution_without_comments)

        try:
            passes = self.run_test_on_gt_solution()
        except Exception as exc:
            raise ValueError(
                "ground-truth solution raised an unexpected test error"
            ) from exc

        if not passes:
            raise ValueError("ground-truth solution does not pass its tests")
        self.validated = True
        return self

    def run_test(self, code: str) -> bool:
        try:
            exec(merge_code_components(code, self.test_code), {})  # noqa: S102
        except AssertionError:
            return False
        return True

    def run_test_on_gt_solution(self) -> bool:
        return self.run_test(self.gt_solution)
