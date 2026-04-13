import ast

from collections.abc import Callable
from typing import Any, Self, cast

from pydantic import BaseModel, Field, model_validator

from nl_code.code_parsing import (
    find_named_assignment_in_body,
    find_named_function,
    get_comments,
    get_docstring,
    literal_eval_assignment_value,
    merge_code_components,
    remove_docstrings_and_comments,
)
from nl_code.datasets.task import Task


def merge_prompt_and_solution(prompt: Any, solution: Any) -> str:
    if not isinstance(prompt, str) or not isinstance(solution, str):
        raise ValueError("prompt and solution must be strings")
    return merge_code_components(prompt, solution)


def extract_prompt_docstring(prompt: Any, entry_point: Any) -> str:
    if not isinstance(prompt, str) or not isinstance(entry_point, str):
        raise ValueError("prompt and entry_point must be strings")
    return get_docstring(find_named_function(prompt, entry_point))


def extract_prompt_comments(prompt: Any) -> str | None:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return get_comments(prompt, strip_hash=True)


def get_check_assignment(test_source: str, name: str, default: object = ...) -> object:
    check_fn = find_named_function(test_source, "check")
    assign = find_named_assignment_in_body(check_fn.body, name)
    if assign is not None:
        return literal_eval_assignment_value(assign)
    if default is ...:
        raise ValueError(f"no `{name} = ...` assignment found inside check()")
    return default


class RawHumanEvalTask(BaseModel):
    task_id: str
    entry_point: str
    prompt: str
    canonical_solution: str
    test: str
    validated: bool = False

    gt_solution: str = Field(
        default_factory=lambda data: merge_prompt_and_solution(
            data.get("prompt"), data.get("canonical_solution")
        )
    )
    gt_solution_without_comments: str = Field(
        default_factory=lambda data: remove_docstrings_and_comments(
            data.get("gt_solution")
        )
    )
    prompt_docstring: str = Field(
        default_factory=lambda data: extract_prompt_docstring(
            data.get("prompt"), data.get("entry_point")
        )
    )
    prompt_comments: str | None = Field(
        default_factory=lambda data: extract_prompt_comments(data.get("prompt"))
    )
    test_inputs: list[Any] = Field(  # ty: ignore[invalid-assignment]
        default_factory=lambda data: get_check_assignment(
            data.get("test"), name="inputs"
        )
    )
    test_results: list[Any] | None = Field(  # ty: ignore[invalid-assignment]
        default_factory=lambda data: get_check_assignment(
            data.get("test"), name="results", default=None
        )
    )

    @model_validator(mode="after")
    def validate_eval_task(self) -> Self:
        if self.validated:
            return self
        if self.test_results is not None and len(self.test_inputs) != len(
            self.test_results
        ):
            raise ValueError("test inputs and results must have the same length")

        find_named_function(self.gt_solution, self.entry_point)
        ast.parse(self.gt_solution_without_comments)
        ast.parse(self.test)

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
        namespace: dict[str, object] = {}
        exec(merge_code_components(code, self.test), namespace)
        try:
            check_fxn = cast(Callable[[object], object], namespace["check"])
            check_fxn(namespace[self.entry_point])
        except AssertionError:
            return False
        return True

    def run_test_on_gt_solution(self) -> bool:
        return self.run_test(self.gt_solution)


class HumanEvalTask(Task):
    def run_test_on_safe_code(self, code: str, raw_task: RawHumanEvalTask) -> bool:
        return raw_task.run_test(code)
