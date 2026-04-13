from pydantic import BaseModel, Field

from nl_code.code_execution.runner import run_assertion_test
from nl_code.code_parsing import remove_docstrings_and_comments
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

    def run_test(self, code: str) -> bool:
        result = run_assertion_test(
            code,
            self.test_code,
            docker_env={"MPLBACKEND": "Agg"},
        )
        return result.passed

    def run_test_on_gt_solution(self) -> bool:
        return self.run_test(self.gt_solution)
