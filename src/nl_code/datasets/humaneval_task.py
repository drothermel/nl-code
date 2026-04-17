import ast
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nl_code.code_parsing import (
    find_named_assignment_in_body,
    find_named_function,
    get_comments,
    get_docstring,
    literal_eval_assignment_value,
    merge_code_components,
    remove_docstrings_and_comments,
)
from nl_code.code_execution.runner import run_assertion_test


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _first_docstring_expr(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Expr | None:
    if not node.body:
        return None

    first_stmt = node.body[0]
    if not isinstance(first_stmt, ast.Expr):
        return None
    if not isinstance(first_stmt.value, ast.Constant):
        return None
    if not isinstance(first_stmt.value.value, str):
        return None
    return first_stmt


def _remove_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source.rstrip() + "\n" if source.strip() else ""
    line_numbers_to_remove: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            docstring_expr = _first_docstring_expr(node)
            if docstring_expr is None:
                continue
            assert docstring_expr.lineno is not None
            assert docstring_expr.end_lineno is not None
            line_numbers_to_remove.update(
                range(docstring_expr.lineno, docstring_expr.end_lineno + 1)
            )

    remaining_lines = [
        line
        for line_number, line in enumerate(source.splitlines(keepends=True), start=1)
        if line_number not in line_numbers_to_remove
    ]
    if not remaining_lines:
        return ""

    return "".join(remaining_lines).rstrip() + "\n"


def build_function_source(prompt: Any, solution: Any) -> str:
    prompt_str = _require_string(prompt, name="prompt")
    solution_str = _require_string(solution, name="solution")
    return merge_code_components(prompt_str, solution_str)


def build_function_without_comments(prompt: Any, solution: Any) -> str:
    function_with_comments = build_function_source(prompt, solution)
    return remove_docstrings_and_comments(function_with_comments)


def extract_docstrings(prompt: Any, entry_point: Any) -> str:
    prompt_str = _require_string(prompt, name="prompt")
    entry_point_str = _require_string(entry_point, name="entry_point")
    return get_docstring(find_named_function(prompt_str, entry_point_str))


def extract_prompt_comment(prompt: Any) -> str:
    prompt_str = _require_string(prompt, name="prompt")
    return get_comments(prompt_str, strip_hash=True) or ""


def build_function_stub(prompt: Any) -> str:
    prompt_str = _require_string(prompt, name="prompt")
    return _remove_docstrings(prompt_str)


def build_assertion_test_code(test_source: Any, entry_point: Any) -> str:
    test_source_str = _require_string(test_source, name="test_source")
    entry_point_str = _require_string(entry_point, name="entry_point")
    return f"{test_source_str}\n\ncheck({entry_point_str})\n"


def build_official_prompt(prompt: Any) -> str:
    prompt_str = _require_string(prompt, name="prompt")
    return (
        "Read the following function signature and docstring, and fully "
        "implement the function described. Your response should only contain "
        "the code for this function.\n\n"
        "```python\n"
        f"{prompt_str.rstrip()}\n"
        "```\n"
    )


def get_check_assignment(test_source: Any, name: str, default: object = ...) -> object:
    test_source_str = _require_string(test_source, name="test_source")
    check_fn = find_named_function(test_source_str, "check")
    assign = find_named_assignment_in_body(check_fn.body, name)
    if assign is not None:
        return literal_eval_assignment_value(assign)
    if default is ...:
        raise ValueError(f"no `{name} = ...` assignment found inside check()")
    return default


class RawHumanEvalTask(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    non_code_fields: ClassVar[tuple[str, ...]] = (
        "docstrings",
        "entry_point",
        "prompt_comment",
        "task_id",
        "validated",
        "version",
    )

    task_id: str
    entry_point: str
    source__prompt: str = Field(alias="prompt")
    source__canonical_solution: str = Field(alias="canonical_solution")
    source__test: str = Field(alias="test")
    version: Literal["v1", "v2"] = "v2"
    validated: bool = False

    official_prompt: str = Field(
        default_factory=lambda data: build_official_prompt(data.get("source__prompt"))
    )
    new_official_prompt: str = Field(
        default_factory=lambda data: data.get("official_prompt")
    )
    docstrings: str = Field(
        default_factory=lambda data: extract_docstrings(
            data.get("source__prompt"),
            data.get("entry_point"),
        )
    )
    prompt_comment: str = Field(
        default_factory=lambda data: extract_prompt_comment(data.get("source__prompt"))
    )
    function_stub: str = Field(
        default_factory=lambda data: build_function_stub(data.get("source__prompt"))
    )
    function_stub_with_comments: str = Field(
        default_factory=lambda data: data.get("source__prompt")
    )
    new_code_stub: str = Field(default_factory=lambda data: data.get("function_stub"))
    new_code_stub_with_comments: str = Field(
        default_factory=lambda data: data.get("function_stub_with_comments")
    )
    function_with_comments: str = Field(
        default_factory=lambda data: build_function_source(
            data.get("source__prompt"),
            data.get("source__canonical_solution"),
        )
    )
    function: str = Field(
        default_factory=lambda data: build_function_without_comments(
            data.get("source__prompt"),
            data.get("source__canonical_solution"),
        )
    )
    gt_solution_with_comments: str = Field(
        default_factory=lambda data: data.get("function_with_comments")
    )
    gt_solution: str = Field(
        default_factory=lambda data: remove_docstrings_and_comments(
            data.get("gt_solution_with_comments")
        )
    )
    assertion_test_code: str = Field(
        default_factory=lambda data: build_assertion_test_code(
            data.get("source__test"),
            data.get("entry_point"),
        )
    )
    test_inputs: list[Any] = Field(  # ty: ignore[invalid-assignment]
        default_factory=lambda data: get_check_assignment(
            data.get("source__test"), name="inputs"
        )
    )
    test_results: list[Any] | None = Field(  # ty: ignore[invalid-assignment]
        default_factory=lambda data: get_check_assignment(
            data.get("source__test"), name="results", default=None
        )
    )

    @model_validator(mode="after")
    def validate_eval_task(self) -> Self:
        if self.test_results is not None and len(self.test_inputs) != len(
            self.test_results
        ):
            raise ValueError("test inputs and results must have the same length")
        return self

    def run_test(self, code: str) -> bool:
        result = run_assertion_test(code, self.assertion_test_code)
        return result.passed

    def run_test_on_gt_solution(self) -> bool:
        return self.run_test(self.gt_solution)
