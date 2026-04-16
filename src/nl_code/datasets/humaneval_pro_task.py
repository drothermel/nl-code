from pydantic import BaseModel, ConfigDict, Field

from nl_code.code_execution.runner import run_assertion_test
from nl_code.code_parsing import remove_docstrings_and_comments
from nl_code.datasets.pro_task_helpers import (
    build_function_stub_without_docstrings,
    build_gt_solution,
    build_new_function_source,
    build_new_function_stub,
    build_new_two_part_function_stub,
    build_problem_stub_without_docstrings_and_comments,
    build_two_part_prompt,
    extract_new_entry_point,
    extract_function_docstring,
    extract_problem_comments,
    extract_source_imports,
    extract_verified_new_docstring,
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

    source__new_function: str = Field(
        default_factory=lambda data: build_new_function_source(
            data.get("source__new_problem"),
            data.get("source__new_solution"),
        )
    )
    source__raw_problem_imports: str = Field(
        default_factory=lambda data: extract_source_imports(
            data.get("source__raw_problem"),
            field_name="source__raw_problem",
        )
    )
    source__new_problem_without_docstrings_and_comments: str = Field(
        default_factory=lambda data: build_problem_stub_without_docstrings_and_comments(
            data.get("source__new_problem"),
            field_name="source__new_problem",
        )
    )
    original_docstrings: str = Field(
        default_factory=lambda data: extract_function_docstring(
            data.get("source__raw_problem"),
            field_name="source__raw_problem",
        )
    )
    new_docstrings: str = Field(
        default_factory=lambda data: extract_verified_new_docstring(
            data.get("source__new_function"),
            data.get("source__new_solution"),
        )
    )
    new_problem_comment: str = Field(
        default_factory=lambda data: extract_problem_comments(
            data.get("source__new_problem"),
            field_name="source__new_problem",
        )
    )
    original_function_stub: str = Field(
        default_factory=lambda data: build_function_stub_without_docstrings(
            data.get("source__raw_problem"),
            field_name="source__raw_problem",
        )
    )
    original_function_stub_with_comments: str = Field(
        default_factory=lambda data: data.get("source__raw_problem")
    )
    new_function_stub: str = Field(
        default_factory=lambda data: build_new_function_stub(
            data.get("source__raw_problem_imports"),
            data.get("source__new_problem_without_docstrings_and_comments"),
        )
    )
    new_function_stub_with_comments: str = Field(
        default_factory=lambda data: data.get("source__new_problem")
    )
    new_two_part_function_stub: str = Field(
        default_factory=lambda data: build_new_two_part_function_stub(
            data.get("source__raw_problem"),
            data.get("source__new_problem_without_docstrings_and_comments"),
        )
    )
    new_two_part_function_stub_with_comments: str = Field(
        default_factory=lambda data: build_two_part_prompt(
            data.get("source__raw_problem"),
            data.get("source__new_problem"),
        )
    )
    original_official_prompt: str = Field(
        default_factory=lambda data: data.get("source__raw_problem")
    )
    new_official_prompt: str = Field(
        default_factory=lambda data: data.get("source__new_problem")
    )
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
        default_factory=lambda data: data.get("new_problem_comment")
    )

    def run_test(self, code: str) -> bool:
        result = run_assertion_test(code, self.source__test_code)
        return result.passed

    def run_test_on_gt_solution(self) -> bool:
        return self.run_test(self.gt_solution)
