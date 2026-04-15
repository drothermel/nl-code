from pydantic import BaseModel, ConfigDict, Field

from nl_code.code_execution.runner import run_assertion_test
from nl_code.code_parsing import remove_docstrings_and_comments
from nl_code.datasets.pro_task_helpers import (
    build_gt_solution,
    extract_new_description,
    extract_new_entry_point,
)


class RawHumanEvalProTask(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str
    source__raw_problem: str = Field(alias="raw_problem")
    source__raw_solution: str = Field(alias="raw_solution")
    source__new_problem: str = Field(alias="new_problem")
    source__new_solution: str = Field(alias="new_solution")
    source__test_code: str = Field(alias="test_code")
    validated: bool = False

    gt_solution: str = Field(
        default_factory=lambda data: build_gt_solution(
            data.get("source__raw_problem"),
            data.get("source__raw_solution"),
            data.get("source__new_problem"),
            data.get("source__new_solution"),
        )
    )
    gt_solution_without_comments: str = Field(
        default_factory=lambda data: remove_docstrings_and_comments(
            data.get("gt_solution")
        )
    )
    new_entry_point: str = Field(
        default_factory=lambda data: extract_new_entry_point(
            data.get("source__new_problem"), data.get("source__new_solution")
        )
    )
    new_description: str = Field(
        default_factory=lambda data: extract_new_description(
            data.get("source__new_problem")
        )
    )

    def run_test(self, code: str) -> bool:
        result = run_assertion_test(code, self.source__test_code)
        return result.passed

    def run_test_on_gt_solution(self) -> bool:
        return self.run_test(self.gt_solution)
