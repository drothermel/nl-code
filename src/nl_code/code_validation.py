import time

from pydantic import BaseModel, Field

from nl_code.code_analysis import (
    check_function_exists,
    check_python_syntax,
    extract_from_code_fences,
)
from nl_code.code_execution.models import TestCase, TestCaseResult
from nl_code.code_execution.runner import EXEC_MODE_DOCKER, run_test_cases


class ValidationResult(BaseModel):
    """Result of validating generated code through the full pipeline."""

    raw_output: str
    extracted_code: str
    had_code_fences: bool
    is_valid_syntax: bool
    syntax_error: str | None = None
    has_expected_function: bool | None = None
    test_case_results: list[TestCaseResult] = Field(default_factory=list)
    test_pass_rate: float | None = None
    elapsed_seconds: float | None = None


def validate_generated_code(
    raw_output: str,
    function_name: str,
    test_cases: list[TestCase] | None = None,
    timeout_seconds: float = 30.0,
    *,
    execution_mode: str = EXEC_MODE_DOCKER,
    docker_image: str | None = None,
) -> ValidationResult:
    """Validate generated Python code through the full pipeline.

    Steps:
    1. Extract code from markdown fences (if present)
    2. Check Python syntax
    3. Verify expected function exists
    4. Run test cases (if provided)
    5. Compute pass rate
    """
    t0 = time.monotonic()

    extracted, had_fences = extract_from_code_fences(raw_output)
    is_valid, syntax_error = check_python_syntax(extracted)

    has_function: bool | None = None
    if is_valid:
        has_function = check_function_exists(extracted, function_name)

    tc_results: list[TestCaseResult] = []
    pass_rate: float | None = None
    if is_valid and has_function and test_cases:
        tc_results, pass_rate = run_test_cases(
            extracted,
            function_name,
            test_cases,
            timeout_seconds,
            execution_mode=execution_mode,
            docker_image=docker_image,
        )

    return ValidationResult(
        raw_output=raw_output,
        extracted_code=extracted,
        had_code_fences=had_fences,
        is_valid_syntax=is_valid,
        syntax_error=syntax_error,
        has_expected_function=has_function,
        test_case_results=tc_results,
        test_pass_rate=pass_rate,
        elapsed_seconds=time.monotonic() - t0,
    )
