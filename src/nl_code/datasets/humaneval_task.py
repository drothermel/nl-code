import ast
from collections.abc import Iterator
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


def strip_surrounding_empty_lines(value: str) -> str:
    lines = value.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


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
    return _require_string(prompt, name="prompt")


def get_check_assignment(test_source: Any, name: str, default: object = ...) -> object:
    test_source_str = _require_string(test_source, name="test_source")
    check_fn = find_named_function(test_source_str, "check")
    assign = find_named_assignment_in_body(check_fn.body, name)
    if assign is not None:
        return literal_eval_assignment_value(assign)
    if default is ...:
        raise ValueError(f"no `{name} = ...` assignment found inside check()")
    return default


HumanEvalTestShape = Literal["inputs_results", "inputs_ref_func"]


class HumanEvalTestCase(BaseModel):
    """A single parsed HumanEval test case."""

    index: int
    input_value: Any
    expected_output: Any | None = None
    has_expected_output: bool = False


def _line_col_to_index(source: str, line_number: int, col_offset: int) -> int:
    lines = source.splitlines(keepends=True)
    line = lines[line_number - 1]
    col_text = line.encode("utf-8")[:col_offset].decode("utf-8")
    return sum(len(line) for line in lines[: line_number - 1]) + len(col_text)


def _node_span(source: str, node: ast.AST) -> tuple[int, int]:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    if (
        not isinstance(lineno, int)
        or not isinstance(col_offset, int)
        or not isinstance(end_lineno, int)
        or not isinstance(end_col_offset, int)
    ):
        raise ValueError(f"{type(node).__name__} node does not have source positions")
    return (
        _line_col_to_index(source, lineno, col_offset),
        _line_col_to_index(source, end_lineno, end_col_offset),
    )


def _replace_source_spans(
    source: str,
    replacements: list[tuple[tuple[int, int], str]],
) -> str:
    updated = source
    for (start, end), replacement in sorted(replacements, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _literal_check_assignment(
    check_fn: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> tuple[ast.Assign, list[Any], list[ast.expr]]:
    assign = find_named_assignment_in_body(check_fn.body, name)
    if assign is None:
        raise ValueError(f"no `{name} = ...` assignment found inside check()")
    if not isinstance(assign.value, ast.List):
        raise TypeError(f"`{name}` assignment must be a list literal")
    value = literal_eval_assignment_value(assign)
    if not isinstance(value, list):
        raise TypeError(f"`{name}` assignment must evaluate to a list")
    return assign, value, assign.value.elts


def _single_item_list_source(source: str, item_node: ast.AST) -> str:
    start, end = _node_span(source, item_node)
    return f"[{source[start:end]}]"


def _check_references_name(
    check_fn: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == name for node in ast.walk(check_fn)
    )


def _normalize_sequence_index(index: int, size: int, *, collection_name: str) -> int:
    normalized_index = index + size if index < 0 else index
    if normalized_index < 0 or normalized_index >= size:
        raise IndexError(
            f"{collection_name} index {index} out of range for {size} items"
        )
    return normalized_index


class HumanEvalTest(BaseModel):
    """A parsed HumanEval test suite preserving the original source shape."""

    source__test: str
    entry_point: str
    shape: HumanEvalTestShape
    inputs: list[Any]
    results: list[Any] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.shape == "inputs_results":
            if self.results is None:
                raise ValueError("inputs_results test shape requires results")
            if len(self.inputs) != len(self.results):
                raise ValueError("test inputs and results must have the same length")
            return self
        if self.results is not None:
            raise ValueError("inputs_ref_func test shape must not define results")
        return self

    @property
    def case_count(self) -> int:
        return len(self.inputs)

    def case_at_index(self, index: int) -> HumanEvalTestCase:
        normalized_index = _normalize_sequence_index(
            index,
            len(self.inputs),
            collection_name="test case",
        )
        if self.results is None:
            return HumanEvalTestCase(
                index=normalized_index,
                input_value=self.inputs[normalized_index],
            )
        return HumanEvalTestCase(
            index=normalized_index,
            input_value=self.inputs[normalized_index],
            expected_output=self.results[normalized_index],
            has_expected_output=True,
        )

    def iter_cases(self) -> Iterator[HumanEvalTestCase]:
        for index in range(len(self.inputs)):
            yield self.case_at_index(index)

    def source_for_index(self, index: int) -> str:
        case = self.case_at_index(index)
        test_source_str = self.source__test
        check_fn = find_named_function(test_source_str, "check")
        inputs_assign, inputs, input_nodes = _literal_check_assignment(
            check_fn,
            "inputs",
        )
        if len(inputs) != len(self.inputs) or len(input_nodes) != len(self.inputs):
            raise ValueError("parsed inputs do not match serialized test suite")

        inputs_span = _node_span(test_source_str, inputs_assign.value)
        replacements = [
            (
                inputs_span,
                _single_item_list_source(test_source_str, input_nodes[case.index]),
            )
        ]
        if self.shape == "inputs_results":
            results_assign, results, result_nodes = _literal_check_assignment(
                check_fn,
                "results",
            )
            if (
                self.results is None
                or len(results) != len(self.results)
                or len(result_nodes) != len(self.results)
            ):
                raise ValueError("parsed results do not match serialized test suite")
            replacements.append(
                (
                    _node_span(test_source_str, results_assign.value),
                    _single_item_list_source(test_source_str, result_nodes[case.index]),
                )
            )
        return _replace_source_spans(test_source_str, replacements)

    def assertion_test_code(self) -> str:
        return build_assertion_test_code(self.source__test, self.entry_point)

    def assertion_test_code_for_index(self, index: int) -> str:
        return build_assertion_test_code(self.source_for_index(index), self.entry_point)

    def run_test(self, code: str) -> bool:
        result = run_assertion_test(code, self.assertion_test_code())
        return result.passed


def parse_inputs_results_test(
    test_source: Any,
    entry_point: Any,
) -> HumanEvalTest:
    test_source_str = _require_string(test_source, name="test_source")
    entry_point_str = _require_string(entry_point, name="entry_point")
    check_fn = find_named_function(test_source_str, "check")
    _inputs_assign, inputs, _input_nodes = _literal_check_assignment(check_fn, "inputs")
    _results_assign, results, _result_nodes = _literal_check_assignment(
        check_fn, "results"
    )
    if len(inputs) != len(results):
        raise ValueError("test inputs and results must have the same length")
    return HumanEvalTest(
        source__test=test_source_str,
        entry_point=entry_point_str,
        shape="inputs_results",
        inputs=inputs,
        results=results,
    )


def parse_inputs_ref_func_test(
    test_source: Any,
    entry_point: Any,
) -> HumanEvalTest:
    test_source_str = _require_string(test_source, name="test_source")
    entry_point_str = _require_string(entry_point, name="entry_point")
    check_fn = find_named_function(test_source_str, "check")
    _inputs_assign, inputs, _input_nodes = _literal_check_assignment(check_fn, "inputs")
    if find_named_assignment_in_body(check_fn.body, "results") is not None:
        raise ValueError("inputs/ref_func test shape must not define `results`")
    if not _check_references_name(check_fn, "ref_func"):
        raise ValueError("inputs/ref_func test shape must reference `ref_func`")
    return HumanEvalTest(
        source__test=test_source_str,
        entry_point=entry_point_str,
        shape="inputs_ref_func",
        inputs=inputs,
        results=None,
    )


def parse_humaneval_test(test_source: Any, entry_point: Any) -> HumanEvalTest:
    test_source_str = _require_string(test_source, name="test_source")
    check_fn = find_named_function(test_source_str, "check")
    if find_named_assignment_in_body(check_fn.body, "results") is not None:
        return parse_inputs_results_test(test_source_str, entry_point)
    return parse_inputs_ref_func_test(test_source_str, entry_point)


def iter_inputs_results_test_cases(
    test_source: Any,
    entry_point: Any,
) -> Iterator[HumanEvalTestCase]:
    return parse_inputs_results_test(test_source, entry_point).iter_cases()


def iter_inputs_ref_func_test_cases(
    test_source: Any,
    entry_point: Any,
) -> Iterator[HumanEvalTestCase]:
    return parse_inputs_ref_func_test(test_source, entry_point).iter_cases()


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
    test_suite: HumanEvalTest
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

    @model_validator(mode="before")
    @classmethod
    def normalize_raw_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        prompt = data.get("prompt", data.get("source__prompt"))
        if isinstance(prompt, str):
            normalized["official_prompt"] = prompt
            normalized["new_official_prompt"] = prompt
        if "test_suite" not in normalized:
            test_source = data.get("test", data.get("source__test"))
            entry_point = data.get("entry_point")
            if isinstance(test_source, str) and isinstance(entry_point, str):
                normalized["test_suite"] = parse_humaneval_test(
                    test_source,
                    entry_point,
                )
        return normalized

    @model_validator(mode="after")
    def validate_eval_task(self) -> Self:
        if self.test_suite.entry_point != self.entry_point:
            raise ValueError("test suite entry point must match task entry point")
        return self

    @property
    def source__test(self) -> str:
        return self.test_suite.source__test

    def _display_(self) -> Any:
        model_dump = self.model_dump()
        try:
            import marimo as mo
        except ImportError:
            return model_dump

        model_name = type(self).__name__
        return mo.vstack(
            [
                mo.md(f"**{model_name}** - {self.task_id}"),
                mo.md("Prompt"),
                mo.ui.code_editor(
                    value=strip_surrounding_empty_lines(self.source__prompt),
                    language="python",
                    disabled=True,
                    min_height=1,
                ),
                mo.md("Canonical Solution"),
                mo.ui.code_editor(
                    value=strip_surrounding_empty_lines(
                        self.source__canonical_solution
                    ),
                    language="python",
                    disabled=True,
                    min_height=1,
                ),
            ]
        )
