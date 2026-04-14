from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nl_code.code_execution.models import UnittestResult
from nl_code.code_execution.runner import run_unittest_test


class MethodDependencies(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    standalone: bool = Field(alias="Standalone")
    lib_dependencies: list[str]
    field_dependencies: list[str]
    method_dependencies: list[str]


class MethodInfo(BaseModel):
    method_name: str
    method_description: str
    solution_code: str
    test_class: str
    test_code: str
    dependencies: MethodDependencies


class ClassEvalTestDetail(BaseModel):
    """Result of running a single unittest test class."""

    __test__ = False

    test_class_name: str
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_errored: int
    failures: list[str]
    errors: list[str]
    passed: bool


class ClassEvalTestResult(BaseModel):
    """Aggregated result across all test classes for a ClassEval task."""

    __test__ = False

    all_passed: bool
    total_tests_run: int
    total_tests_passed: int
    total_tests_failed: int
    total_tests_errored: int
    per_test_class: list[ClassEvalTestDetail]
    error: str | None = None


def _build_gt_code(import_statement: Any, solution_code: Any) -> str:
    if not isinstance(solution_code, str):
        raise ValueError("solution_code must be a string")
    if not isinstance(import_statement, list):
        raise ValueError("import_statement must be a list")
    imports = "\n".join(import_statement)
    if imports:
        return imports + "\n\n" + solution_code
    return solution_code


# ---------------------------------------------------------------------------
# Per-task fixes for dataset issues.
#
# Each entry maps a task_id to a dict with optional keys:
#   "solution": list of (old, new) string replacements on solution_code
#   "test":     list of (old, new) string replacements on test code
#   "test_classes_remove": list of test class names to drop
#   "reason":   human-readable explanation
# ---------------------------------------------------------------------------
_TASK_FIXES: dict[str, dict[str, Any]] = {
    "ClassEval_17": {
        "test": [
            # get_upcoming_events uses datetime.now(); 2024 dates are now in the past
            ("datetime(2024,", "datetime(2099,"),
        ],
        "reason": "Test dates in the past due to datetime.now() comparison",
    },
    "ClassEval_31": {
        "test": [
            # Float comparison at 1e-16 precision fails with assertEqual
            (
                "self.assertEqual(DataStatistics4.correlation_coefficient",
                "self.assertAlmostEqual(DataStatistics4.correlation_coefficient",
            ),
        ],
        "reason": "Float equality fails at 1e-16 precision",
    },
    "ClassEval_48": {
        "test": [
            # Hostname is machine-specific, not portable
            (
                "self.assertEqual(result, 'LAPTOP-2CS86KUM')",
                "self.assertIsInstance(result, str)",
            ),
        ],
        "reason": "Hardcoded hostname is machine-specific",
    },
    "ClassEval_51": {
        "solution": [
            # np.mat was removed in NumPy 2.0
            ("np.mat(", "np.asmatrix("),
            # float() on 1x1 matrix no longer works in NumPy 2.0
            ("float(ysum * xsum)", "float((ysum * xsum).item())"),
        ],
        "test": [
            # NumPy 2.0 returns np.float64 which differs at last decimal
            (
                "self.assertEqual(KappaCalculator.fleiss_kappa",
                "self.assertAlmostEqual(KappaCalculator.fleiss_kappa",
            ),
        ],
        "reason": "np.mat() removed and float() on 1x1 matrix broken in NumPy 2.0",
    },
    "ClassEval_58": {
        "auto_fail": True,
        "reason": "Solution bug: random mine placement can overlap, causing "
        "nondeterministic test failures (~20% of runs)",
    },
    "ClassEval_69": {
        "test_classes_remove": ["TestPDFHandler"],
        "reason": "TestPDFHandler is a fixture class with no test methods",
    },
}


def _apply_fixes(
    task_id: str,
    solution_code: str,
    test: str,
    test_classes: list[str],
) -> tuple[str, str, list[str], bool, bool, str | None]:
    """Apply hardcoded fixes.

    Returns (solution, test, test_classes, pp_solution, pp_test, auto_fail_reason).
    """
    fixes = _TASK_FIXES.get(task_id)
    if fixes is None:
        return solution_code, test, test_classes, False, False, None

    if fixes.get("auto_fail"):
        return solution_code, test, test_classes, False, False, fixes["reason"]

    pp_solution = False
    pp_test = False

    for old, new in fixes.get("solution", []):
        solution_code = solution_code.replace(old, new)
        pp_solution = True

    for old, new in fixes.get("test", []):
        test = test.replace(old, new)
        pp_test = True

    removals = fixes.get("test_classes_remove", [])
    if removals:
        test_classes = [tc for tc in test_classes if tc not in removals]
        pp_test = True

    return solution_code, test, test_classes, pp_solution, pp_test, None


class RawClassEvalTask(BaseModel):
    task_id: str
    class_name: str
    class_description: str
    class_constructor: str
    fields: list[str]
    import_statement: list[str]
    skeleton: str
    solution_code: str
    test: str
    test_classes: list[str]
    methods_info: list[MethodInfo]
    validated: bool = False
    postprocess_solution: bool = False
    postprocess_test: bool = False
    auto_fail_reason: str | None = None

    gt_code: str = Field(
        default_factory=lambda data: _build_gt_code(
            data.get("import_statement"), data.get("solution_code")
        )
    )

    @model_validator(mode="after")
    def validate_eval_task(self) -> Self:
        (
            self.solution_code,
            self.test,
            self.test_classes,
            self.postprocess_solution,
            self.postprocess_test,
            self.auto_fail_reason,
        ) = _apply_fixes(self.task_id, self.solution_code, self.test, self.test_classes)

        # Recompute gt_code if solution was modified
        if self.postprocess_solution:
            self.gt_code = _build_gt_code(self.import_statement, self.solution_code)
        return self

    def run_test(self, code: str) -> ClassEvalTestResult:
        result = run_unittest_test(code, self.test, self.test_classes)
        return _class_eval_result_from_unittest_result(result)

    def run_test_on_gt_solution(self) -> ClassEvalTestResult:
        return self.run_test(self.gt_code)


def _class_eval_result_from_unittest_result(
    result: UnittestResult,
) -> ClassEvalTestResult:
    return ClassEvalTestResult(
        all_passed=result.all_passed,
        total_tests_run=result.total_tests_run,
        total_tests_passed=result.total_tests_passed,
        total_tests_failed=result.total_tests_failed,
        total_tests_errored=result.total_tests_errored,
        per_test_class=[
            ClassEvalTestDetail(
                test_class_name=detail.test_class_name,
                tests_run=detail.tests_run,
                tests_passed=detail.tests_passed,
                tests_failed=detail.tests_failed,
                tests_errored=detail.tests_errored,
                failures=detail.failures,
                errors=detail.errors,
                passed=detail.passed,
            )
            for detail in result.per_test_class
        ],
        error=result.error,
    )
