import ast
from collections.abc import Iterator
from functools import cached_property
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nl_code.code_parsing import (
    find_named_assignment_in_body,
    find_named_function,
    literal_eval_assignment_value,
    merge_code_components,
    node_references_name,
    node_span,
    remove_docstrings_and_comments,
    replace_source_spans,
)
from nl_code.code_execution.runner import run_assertion_test
from nl_code.datasets.text import strip_surrounding_empty_lines
from nl_code.datasets.validation import require_string


def build_function_source(prompt: Any, solution: Any) -> str:
    prompt_str = require_string(prompt, name="prompt")
    solution_str = require_string(solution, name="solution")
    return merge_code_components(prompt_str, solution_str)


def build_assertion_test_code(test_source: Any, entry_point: Any) -> str:
    test_source_str = require_string(test_source, name="test_source")
    entry_point_str = require_string(entry_point, name="entry_point")
    return f"{test_source_str}\n\ncheck({entry_point_str})\n"


def build_official_prompt(prompt: Any) -> str:
    return require_string(prompt, name="prompt")


def get_check_assignment(test_source: Any, name: str, default: object = ...) -> object:
    test_source_str = require_string(test_source, name="test_source")
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
    start, end = node_span(source, item_node)
    return f"[{source[start:end]}]"


def _normalize_sequence_index(index: int, size: int, *, collection_name: str) -> int:
    normalized_index = index + size if index < 0 else index
    if normalized_index < 0 or normalized_index >= size:
        raise IndexError(
            f"{collection_name} index {index} out of range for {size} items"
        )
    return normalized_index


class HumanEvalTest(BaseModel):
    """A parsed HumanEval test suite preserving the original source shape."""

    source: str
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
        test_source_str = self.source
        check_fn = find_named_function(test_source_str, "check")
        inputs_assign, inputs, input_nodes = _literal_check_assignment(
            check_fn,
            "inputs",
        )
        if len(inputs) != len(self.inputs) or len(input_nodes) != len(self.inputs):
            raise ValueError("parsed inputs do not match serialized test suite")

        inputs_span = node_span(test_source_str, inputs_assign.value)
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
                    node_span(test_source_str, results_assign.value),
                    _single_item_list_source(test_source_str, result_nodes[case.index]),
                )
            )
        return replace_source_spans(test_source_str, replacements)

    def assertion_test_code(self) -> str:
        return build_assertion_test_code(self.source, self.entry_point)

    def assertion_test_code_for_index(self, index: int) -> str:
        return build_assertion_test_code(self.source_for_index(index), self.entry_point)

    def run_test(self, code: str) -> bool:
        result = run_assertion_test(code, self.assertion_test_code())
        return result.passed


def parse_inputs_results_test(
    test_source: Any,
    entry_point: Any,
) -> HumanEvalTest:
    test_source_str = require_string(test_source, name="test_source")
    entry_point_str = require_string(entry_point, name="entry_point")
    check_fn = find_named_function(test_source_str, "check")
    _inputs_assign, inputs, _input_nodes = _literal_check_assignment(check_fn, "inputs")
    _results_assign, results, _result_nodes = _literal_check_assignment(
        check_fn, "results"
    )
    if len(inputs) != len(results):
        raise ValueError("test inputs and results must have the same length")
    return HumanEvalTest(
        source=test_source_str,
        entry_point=entry_point_str,
        shape="inputs_results",
        inputs=inputs,
        results=results,
    )


def parse_inputs_ref_func_test(
    test_source: Any,
    entry_point: Any,
) -> HumanEvalTest:
    test_source_str = require_string(test_source, name="test_source")
    entry_point_str = require_string(entry_point, name="entry_point")
    check_fn = find_named_function(test_source_str, "check")
    _inputs_assign, inputs, _input_nodes = _literal_check_assignment(check_fn, "inputs")
    if find_named_assignment_in_body(check_fn.body, "results") is not None:
        raise ValueError("inputs/ref_func test shape must not define `results`")
    if not node_references_name(check_fn, "ref_func"):
        raise ValueError("inputs/ref_func test shape must reference `ref_func`")
    return HumanEvalTest(
        source=test_source_str,
        entry_point=entry_point_str,
        shape="inputs_ref_func",
        inputs=inputs,
        results=None,
    )


def parse_humaneval_test(test_source: Any, entry_point: Any) -> HumanEvalTest:
    test_source_str = require_string(test_source, name="test_source")
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


class HumanEvalSource(BaseModel):
    prompt: str
    canonical_solution: str
    test: str


class RawHumanEvalTask(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    non_code_fields: ClassVar[tuple[str, ...]] = (
        "entry_point",
        "task_id",
        "validated",
        "version",
    )

    task_id: str
    entry_point: str
    source: HumanEvalSource
    version: Literal["v1", "v2"] = "v2"
    validated: bool = False

    gt_solution_with_comments: str = Field(
        default_factory=lambda data: build_function_source(
            data["source"].prompt,
            data["source"].canonical_solution,
        )
    )
    gt_solution: str = Field(
        default_factory=lambda data: remove_docstrings_and_comments(
            data.get("gt_solution_with_comments")
        )
    )

    @model_validator(mode="after")
    def validate_eval_task(self) -> Self:
        parse_humaneval_test(self.source.test, self.entry_point)
        return self

    @cached_property
    def test_suite(self) -> HumanEvalTest:
        return parse_humaneval_test(self.source.test, self.entry_point)

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
                    value=strip_surrounding_empty_lines(self.source.prompt),
                    language="python",
                    disabled=True,
                    min_height=1,
                ),
                mo.md("Canonical Solution"),
                mo.ui.code_editor(
                    value=strip_surrounding_empty_lines(self.source.canonical_solution),
                    language="python",
                    disabled=True,
                    min_height=1,
                ),
            ]
        )
