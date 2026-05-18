from functools import cached_property
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from nl_code.code_execution.runner import run_assertion_test
from nl_code.code_parsing import remove_docstrings_and_comments
from nl_code.datasets.pro_task_helpers import (
    build_function_stub_without_docstrings_and_comments,
    build_gt_solution,
    build_new_official_prompt,
    build_new_function_source,
    build_new_function_without_docstrings_and_comments,
    build_new_function_stub,
    build_new_two_part_function_stub,
    build_original_official_prompt,
    build_original_function_source,
    build_original_function_without_docstrings_and_comments,
    build_problem_stub_without_docstrings_and_comments,
    build_two_part_code,
    build_two_part_prompt,
    extract_docstrings_and_comments,
    extract_new_entry_point,
    extract_problem_comments,
    extract_source_imports,
    extract_verified_new_docstrings_and_comments,
)
from nl_code.datasets.task import TaskTarget


class ProTaskSource(BaseModel):
    raw_problem: str
    raw_solution: str
    new_problem: str
    new_solution: str
    test_code: str


class ProTaskPrompts(BaseModel):
    source: ProTaskSource

    @cached_property
    def original_official(self) -> str:
        return build_original_official_prompt(self.source.raw_problem)

    @cached_property
    def new_official(self) -> str:
        return build_new_official_prompt(
            self.source.raw_problem,
            self.source.new_problem,
        )


class ProOriginalSolution(BaseModel):
    source: ProTaskSource

    @cached_property
    def code_with_comments(self) -> str:
        return build_original_function_source(
            self.source.raw_problem,
            self.source.raw_solution,
        )

    @cached_property
    def code(self) -> str:
        return build_original_function_without_docstrings_and_comments(
            self.source.raw_problem,
            self.source.raw_solution,
        )

    @cached_property
    def stub_with_comments(self) -> str:
        return self.source.raw_problem

    @cached_property
    def stub(self) -> str:
        return build_function_stub_without_docstrings_and_comments(
            self.source.raw_problem,
            field_name="source.raw_problem",
        )

    @cached_property
    def docstrings_and_comments(self) -> str:
        return extract_docstrings_and_comments(
            self.source.raw_problem,
            field_name="source.raw_problem",
        )

    @cached_property
    def imports(self) -> str:
        return remove_docstrings_and_comments(
            extract_source_imports(
                self.source.raw_problem,
                field_name="source.raw_problem",
            )
        )


class ProNewSolution(BaseModel):
    source: ProTaskSource
    original: ProOriginalSolution

    @cached_property
    def code_with_comments(self) -> str:
        return build_new_function_source(
            self.source.new_problem,
            self.source.new_solution,
        )

    @cached_property
    def code(self) -> str:
        return build_new_function_without_docstrings_and_comments(
            self.source.new_problem,
            self.source.new_solution,
        )

    @cached_property
    def problem_without_docstrings_and_comments(self) -> str:
        return build_problem_stub_without_docstrings_and_comments(
            self.source.new_problem,
            field_name="source.new_problem",
        )

    @cached_property
    def stub_with_comments(self) -> str:
        return self.source.new_problem

    @cached_property
    def stub(self) -> str:
        return build_new_function_stub(
            self.original.imports,
            self.problem_without_docstrings_and_comments,
        )

    @cached_property
    def two_part_stub_with_comments(self) -> str:
        return build_two_part_prompt(
            self.source.raw_problem,
            self.source.new_problem,
        )

    @cached_property
    def two_part_stub(self) -> str:
        return build_new_two_part_function_stub(
            self.original.stub,
            self.problem_without_docstrings_and_comments,
        )

    @cached_property
    def docstrings_and_comments(self) -> str:
        return extract_verified_new_docstrings_and_comments(
            self.source.new_problem,
            self.source.new_solution,
        )

    @cached_property
    def problem_comment(self) -> str:
        return extract_problem_comments(
            self.source.new_problem,
            field_name="source.new_problem",
        )


class ProGTSolution(BaseModel):
    original: ProOriginalSolution
    new: ProNewSolution

    @cached_property
    def code_with_comments(self) -> str:
        source = self.original.source
        return build_gt_solution(
            source.raw_problem,
            source.raw_solution,
            source.new_problem,
            source.new_solution,
        )

    @cached_property
    def code(self) -> str:
        return build_two_part_code(
            self.original.code,
            self.new.code,
        )

    def run_test(self, test_suite: "ProAssertionTestSuite") -> bool:
        return test_suite.run_test(self.code)


class ProAssertionTestSuite(BaseModel):
    source: str
    docker_env: dict[str, str] | None = None

    def run_test(self, code: str) -> bool:
        result = run_assertion_test(code, self.source, docker_env=self.docker_env)
        return result.passed


class RawProTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: ProTaskSource
    version: Literal["v3"] = "v3"
    validated: bool = False

    @model_validator(mode="after")
    def validate_source_derivations(self) -> Self:
        _ = self.new_solution.docstrings_and_comments
        _ = self.target.name
        return self

    @cached_property
    def original_solution(self) -> ProOriginalSolution:
        return ProOriginalSolution(source=self.source)

    @cached_property
    def new_solution(self) -> ProNewSolution:
        return ProNewSolution(
            source=self.source,
            original=self.original_solution,
        )

    @cached_property
    def gt_solution(self) -> ProGTSolution:
        return ProGTSolution(
            original=self.original_solution,
            new=self.new_solution,
        )

    @cached_property
    def prompts(self) -> ProTaskPrompts:
        return ProTaskPrompts(source=self.source)

    @cached_property
    def target(self) -> TaskTarget:
        return TaskTarget(
            name=extract_new_entry_point(
                self.source.new_problem,
                self.source.new_solution,
            ),
            kind="function",
        )

    @cached_property
    def description(self) -> str:
        return self.new_solution.problem_comment

    @cached_property
    def test_suite(self) -> ProAssertionTestSuite:
        return ProAssertionTestSuite(source=self.source.test_code)

    def run_test(self, code: str) -> bool:
        return self.test_suite.run_test(code)

    def run_test_on_gt_solution(self) -> bool:
        return self.gt_solution.run_test(self.test_suite)
