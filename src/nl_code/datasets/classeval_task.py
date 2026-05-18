from functools import cached_property
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from nl_code.code_execution.models import UnittestResult
from nl_code.code_execution.runner import run_unittest_test
from nl_code.code_parsing import (
    remove_docstrings_preserving_comments,
    remove_docstrings_and_comments,
)
from nl_code.datasets.task import TaskTarget
from nl_code.datasets.validation import require_string


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


def _require_string_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list[str]")
    return cast(list[str], list(value))


def _build_import_block(import_statement: Any) -> str:
    imports = _require_string_list(import_statement, name="import_statement")
    if not imports:
        return ""
    return "\n".join(imports) + "\n"


def _build_gt_code(import_statement: Any, solution_code: Any) -> str:
    solution_code_str = require_string(solution_code, name="solution_code")
    imports = _build_import_block(import_statement).rstrip()
    if imports:
        return imports + "\n\n" + solution_code_str
    return solution_code_str


def _build_new_official_prompt(class_name: str, official_skeleton: str) -> str:
    return (
        "Provided below is an instruction detailing a task. Compose a response "
        "that aptly fulfills the request.\n\n"
        f"Please complete the class {class_name} in the subsequent code.\n"
        "```python\n"
        f"{official_skeleton.rstrip()}\n"
        "```"
    )


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
        "auto_fail": True,
        "reason": "Hostname lookup for 0.0.0.0 is environment-dependent; "
        "the dataset's ground truth and tests do not behave consistently "
        "across Docker environments",
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


class ClassEvalSource(BaseModel):
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


class ClassEvalFixedSource(BaseModel):
    solution_code: str
    test: str
    test_classes: list[str]
    postprocess_solution: bool = False
    postprocess_test: bool = False
    auto_fail_reason: str | None = None


class ClassEvalGTSolution(BaseModel):
    source: ClassEvalSource
    fixed: ClassEvalFixedSource

    @cached_property
    def code_with_comments(self) -> str:
        return _build_gt_code(
            self.source.import_statement,
            self.fixed.solution_code,
        )

    @cached_property
    def code(self) -> str:
        return remove_docstrings_and_comments(self.code_with_comments)

    def run_test(self, test_suite: "ClassEvalTestSuite") -> ClassEvalTestResult:
        return test_suite.run_test(self.code)


class ClassEvalPrompt(BaseModel):
    source: ClassEvalSource

    @cached_property
    def new_official(self) -> str:
        return _build_new_official_prompt(
            self.source.class_name,
            self.source.skeleton,
        )


class ClassEvalTestSuite(BaseModel):
    source: str
    test_classes: list[str]

    def run_test(self, code: str) -> ClassEvalTestResult:
        result = run_unittest_test(code, self.source, self.test_classes)
        return _class_eval_result_from_unittest_result(result)


class RawClassEvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: ClassEvalSource
    version: Literal["v3"] = "v3"
    validated: bool = False

    @cached_property
    def fixed_source(self) -> ClassEvalFixedSource:
        (
            solution_code,
            test,
            test_classes,
            postprocess_solution,
            postprocess_test,
            auto_fail_reason,
        ) = _apply_fixes(
            self.task_id,
            self.source.solution_code,
            self.source.test,
            list(self.source.test_classes),
        )
        return ClassEvalFixedSource(
            solution_code=solution_code,
            test=test,
            test_classes=test_classes,
            postprocess_solution=postprocess_solution,
            postprocess_test=postprocess_test,
            auto_fail_reason=auto_fail_reason,
        )

    @cached_property
    def target(self) -> TaskTarget:
        return TaskTarget(name=self.source.class_name, kind="class")

    @cached_property
    def prompt(self) -> ClassEvalPrompt:
        return ClassEvalPrompt(source=self.source)

    @cached_property
    def class_stub_with_comments(self) -> str:
        return self.source.skeleton

    @cached_property
    def class_stub(self) -> str:
        return remove_docstrings_preserving_comments(self.source.skeleton)

    @cached_property
    def import_block(self) -> str:
        return _build_import_block(self.source.import_statement)

    @cached_property
    def gt_solution(self) -> ClassEvalGTSolution:
        return ClassEvalGTSolution(source=self.source, fixed=self.fixed_source)

    @cached_property
    def test_suite(self) -> ClassEvalTestSuite:
        return ClassEvalTestSuite(
            source=self.fixed_source.test,
            test_classes=self.fixed_source.test_classes,
        )

    def run_test(self, code: str) -> ClassEvalTestResult:
        return self.test_suite.run_test(code)

    def run_test_on_gt_solution(self) -> ClassEvalTestResult:
        return self.gt_solution.run_test(self.test_suite)


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
