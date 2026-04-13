from typing import Any

from pydantic import BaseModel, Field

DEFAULT_CODE_EVAL_IMAGE = "nl-code/code-eval:v1"
SCIENTIFIC_CODE_EVAL_IMAGE = "nl-code/code-eval-scientific:v1"


class CodeExecutionInfrastructureError(RuntimeError):
    """Raised when the execution platform itself fails.

    This error means Docker/subprocess infrastructure could not run the code.
    It is NEVER raised for code-level failures (syntax errors, runtime
    exceptions, wrong answers) — those are returned in result objects.

    Callers should ``try/except CodeExecutionInfrastructureError`` to handle
    platform issues and inspect result objects for code-level failures.
    """

    def __init__(
        self,
        *,
        stage: str,
        execution_mode: str,
        detail: str,
    ) -> None:
        self.stage = stage
        self.execution_mode = execution_mode
        self.detail = detail
        super().__init__(
            f"execution infrastructure failure "
            f"(stage={stage}, mode={execution_mode}): {detail}"
        )


# ---------------------------------------------------------------------------
# Function-call execution models
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    """Result of executing a function with a single input."""

    input_value: Any
    return_value: Any | None = None
    return_type: str | None = None
    stdout: str = ""
    stdout_truncated: bool = False
    error: str | None = None
    compile_success: bool | None = None
    compile_error: str | None = None


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
    compile_success: bool | None = None
    compile_error: str | None = None


# ---------------------------------------------------------------------------
# Assertion execution models (Pro datasets)
# ---------------------------------------------------------------------------


class AssertionTestResult(BaseModel):
    """Result of running code against assertion-based test code."""

    __test__ = False

    passed: bool
    error: str | None = None
    stdout: str = ""
    compile_success: bool | None = None
    compile_error: str | None = None


# ---------------------------------------------------------------------------
# Unittest execution models (ClassEval)
# ---------------------------------------------------------------------------


class UnittestTestDetail(BaseModel):
    """Result of running a single unittest.TestCase class."""

    __test__ = False

    test_class_name: str
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_errored: int
    tests_skipped: int = 0
    failures: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    passed: bool


class UnittestResult(BaseModel):
    """Aggregated result across all unittest test classes."""

    all_passed: bool
    total_tests_run: int
    total_tests_passed: int
    total_tests_failed: int
    total_tests_errored: int
    per_test_class: list[UnittestTestDetail]
    error: str | None = None


# ---------------------------------------------------------------------------
# Batch item models
# ---------------------------------------------------------------------------


class FunctionCallBatchItem(BaseModel):
    """A single item for batch_run_test_cases."""

    code: str
    function_name: str
    test_cases: list[TestCase]


class AssertionBatchItem(BaseModel):
    """A single item for batch_run_assertion_tests."""

    code: str
    test_code: str


class UnittestBatchItem(BaseModel):
    """A single item for batch_run_unittest_tests."""

    code: str
    test_code: str
    test_class_names: list[str]
