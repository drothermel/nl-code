from typing import Any

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """Result of executing a function with a single input."""

    input_value: Any
    return_value: Any | None = None
    return_type: str | None = None
    stdout: str = ""
    stdout_truncated: bool = False
    error: str | None = None


class TestCase(BaseModel):
    """A single test case: input and expected output."""

    __test__ = False

    input_value: Any
    expected_output: Any


class TestCaseResult(BaseModel):
    """Result of comparing execution output to expected output."""

    __test__ = False

    input_value: Any
    expected_output: Any
    actual_output: Any | None = None
    passed: bool = False
    error: str | None = None
