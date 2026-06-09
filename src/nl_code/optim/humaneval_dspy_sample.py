from __future__ import annotations

from nl_code.code_execution.models import TestCase
from nl_code.datasets.humaneval_task import RawHumanEvalTask


def code_stub(raw: RawHumanEvalTask) -> str:
    return raw.source.prompt


def function_stub(raw: RawHumanEvalTask) -> str:
    return raw.source.prompt


def gt_code(raw: RawHumanEvalTask) -> str:
    return raw.gt_solution.code


def has_function_call_tests(raw: RawHumanEvalTask) -> bool:
    return raw.test_suite.shape == "inputs_results"


def test_cases(raw: RawHumanEvalTask) -> list[TestCase]:
    if not has_function_call_tests(raw):
        raise ValueError("sample does not provide expected test results")
    results = raw.test_suite.results
    if results is None:
        raise ValueError("sample does not provide expected test results")
    return [
        TestCase(input_value=input_value, expected_output=expected_output)
        for input_value, expected_output in zip(
            raw.test_suite.inputs,
            results,
            strict=True,
        )
    ]


__all__ = [
    "code_stub",
    "function_stub",
    "gt_code",
    "has_function_call_tests",
    "test_cases",
]
